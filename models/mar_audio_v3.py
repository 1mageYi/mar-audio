from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 我们将直接从timm导入一些基础模块，但会自己构建Block

from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Attention as SelfAttention, Block as TimmBlock


import util.misc as misc
from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    """根据给定的顺序和长度创建掩码。"""
    masking = torch.zeros(bsz, seq_len, device=order.device)
    
    # --- 关键修改点 1 ---
    # 使用 .item() 将单元素Tensor转换为Python整数，确保切片操作有效
    mask_len_as_int = mask_len.long().item()
    
    # 使用 scatter 根据顺序和长度填充掩码
    index_to_mask = order[:, :mask_len_as_int]
    masking = torch.scatter(masking, dim=-1, index=index_to_mask, src=torch.ones_like(index_to_mask, dtype=masking.dtype)).bool()
    return masking

class CrossAttention(nn.Module):
    """
    一个标准的交叉注意力模块。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 我们需要为 Q, K, V 创建独立的线性层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        """
        x: 查询 (Query), 来自音频序列, shape [B, L, D]
        context: 键/值 (Key/Value), 来自文本序列, shape [B, T, D]
        """
        B, N, C = x.shape
        B, T, C = context.shape
        
        # 1. 从 x 生成 q
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # 2. 从 context 生成 k 和 v
        k = self.k_proj(context).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 3. 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # 自注意力仍然使用timm的高效实现
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.norm_cross = norm_layer(dim)
        # --- 关键修复点 2: 使用我们新的 CrossAttention 模块 ---
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # ----------------------------------------------------
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, encoder_hidden_states):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.norm_cross(x), encoder_hidden_states))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MAR_Audio(nn.Module):
    """
    适用于音频生成的 MAR 模型 (已重构)。
    - 文本编码器被剥离，本模型直接接收文本特征向量。
    """
    def __init__(self,
                 max_seq_len=1024,
                 audio_embed_dim=64,
                 text_feature_dim=768,
                 encoder_embed_dim=1024,
                 encoder_depth=16,
                 encoder_num_heads=16,
                 decoder_embed_dim=1024,
                 decoder_depth=16,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 mask_ratio_min=0.5,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.audio_embed_dim = audio_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # --- 文本条件处理 ---
        self.text_feature_proj = nn.Linear(text_feature_dim, decoder_embed_dim)
        # null embedding现在需要匹配文本序列的形状，用于CFG
        self.null_text_feature = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --- MAR 编码器 (简化: 只处理音频, 无需buffer) ---
        from timm.models.vision_transformer import Block as TimmBlock
        self.encoder_embed = nn.Linear(audio_embed_dim, encoder_embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            TimmBlock(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --- MAR 解码器 (使用新的 CrossAttentionBlock, 无需buffer) ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
            
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_out_norm = norm_layer(decoder_embed_dim) # 保持输出稳定

        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, max_seq_len, decoder_embed_dim))
        
        self.initialize_weights()

        self.diffloss = DiffLoss(
            target_channels=self.audio_embed_dim, z_channels=decoder_embed_dim, width=diffloss_w, 
            depth=diffloss_d, num_sampling_steps=num_sampling_steps, grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.null_text_feature, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, bsz, seq_len, device):
        """生成随机顺序。"""
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        return torch.tensor(np.array(orders), device=device, dtype=torch.long)

    def random_masking(self, x, orders):
        """根据随机顺序和动态掩码率生成掩码。"""
        bsz, seq_len, _ = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask
    
    def forward_mae_encoder(self, x, mask):
        # x: 完整的音频潜码 [B, L, D_audio]
        # mask: 布尔掩码 [B, L], True代表被掩码
        bsz = x.shape[0]
        x = self.encoder_embed(x)
        x = x + self.encoder_pos_embed[:, :x.shape[1], :]
        
        # 只将未被掩码的token送入encoder
        x = x[~mask].reshape(bsz, -1, self.encoder_embed_dim)
        
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)

    def forward_mae_decoder(self, x_encoded, mask, text_features):
        x = self.decoder_embed(x_encoded)
        
        bsz, seq_len = mask.shape
        decoder_input = self.mask_token.repeat(bsz, seq_len, 1)
        
        # --- 关键修复点 ---
        # 在原地填充之前，将源张量 x 的数据类型转换为与目标张量 decoder_input 一致
        source_tensor = x.reshape(-1, self.decoder_embed_dim).to(decoder_input.dtype)
        decoder_input[~mask] = source_tensor
        # --------------------------------
        
        decoder_input = decoder_input + self.decoder_pos_embed[:, :seq_len, :]

        for blk in self.decoder_blocks:
            decoder_input = blk(decoder_input, text_features)
            
        x_decoded = self.decoder_norm(decoder_input)
        x_decoded = x_decoded + self.diffusion_pos_embed_learned[:, :seq_len, :]
        x_decoded = self.decoder_out_norm(x_decoded)
        return x_decoded
    
    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss
    
    def forward(self, audio_latents, text_features, uncond_prob=0.1):
        # text_features: [B, T, D_text]
        bsz, seq_len, _ = audio_latents.shape
        device = audio_latents.device
        
        projected_text_features = self.text_feature_proj(text_features)
        
        if self.training:
            uncond_mask = (torch.rand(bsz, 1, 1, device=device) < uncond_prob)
            null_text_seq = self.null_text_feature.repeat(1, projected_text_features.shape[1], 1)
            projected_text_features = torch.where(uncond_mask, null_text_seq, projected_text_features)
        
        orders = self.sample_orders(bsz, seq_len, device)
        # 注意: random_masking 返回的是一个值为0/1的浮点数掩码，我们需要布尔型
        mask = self.random_masking(audio_latents, orders).bool()
        gt_latents = audio_latents.clone().detach()

        x_encoded = self.forward_mae_encoder(audio_latents, mask)
        z = self.forward_mae_decoder(x_encoded, mask, projected_text_features)
        
        loss = self.forward_loss(z, gt_latents, mask.float())
        return loss

    @torch.no_grad()
    def sample_tokens(self, text_features, num_iter=64, cfg_scale=4.0, temperature=1.0, progress=True):
        bsz, seq_len_text, _ = text_features.shape
        device = text_features.device
        seq_len_audio = 215

        cond_features = self.text_feature_proj(text_features)
        
        # --- 关键修复点 ---
        # 创建一个与 cond_features 形状完全匹配的 null 特征序列
        # 我们在批次(bsz)和序列(seq_len_text)维度上进行重复
        null_features = self.null_text_feature.repeat(bsz, seq_len_text, 1)
        # ---------------------------------------------------------
        
        mask = torch.ones(bsz, seq_len_audio, device=device)
        tokens = torch.zeros(bsz, seq_len_audio, self.audio_embed_dim, device=device)
        orders = self.sample_orders(bsz, seq_len_audio, device)

        indices = tqdm(range(num_iter)) if progress and misc.is_main_process() else range(num_iter)

        for step in indices:
            # ... (encoder 调用和 z 的计算等逻辑不变) ...
            x_encoded = self.forward_mae_encoder(tokens, mask.bool())
            
            # 准备 CFG 的输入
            full_cond_features = torch.cat([cond_features, null_features], dim=0)
            
            # 为了匹配CFG，我们需要将mask和encoder的输出也复制一份
            doubled_mask = mask.bool().repeat(2, 1)
            doubled_x_encoded = x_encoded.repeat(2, 1, 1)

            z = self.forward_mae_decoder(doubled_x_encoded, doubled_mask, full_cond_features)
            
            z_cond, z_uncond = z.chunk(2)
            z_cfg = z_uncond + cfg_scale * (z_cond - z_uncond)

            # ... (后续的掩码更新和采样逻辑不变) ...
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            target_mask_len = torch.tensor(np.floor(seq_len_audio * mask_ratio), device=device)
            current_mask_len = mask.sum(dim=-1)[0]
            max_len = current_mask_len - 1
            min_len = 1 if step < num_iter - 1 else 0
            next_mask_len = torch.clamp(target_mask_len, min=min_len, max=max_len)
            
            mask_next = mask_by_order(next_mask_len, orders, bsz, seq_len_audio)
            mask_to_pred = (mask.bool() ^ mask_next.bool())
            
            z_pred = z_cfg[mask_to_pred]
            
            sampled_latents = self.diffloss.sample(z_pred, temperature=temperature, cfg=1.0)
            
            tokens[mask_to_pred] = sampled_latents.to(tokens.dtype)
            mask = mask_next

        return tokens

def mar_audio_base(**kwargs):
    model = MAR_Audio(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mar_audio_large(**kwargs):
    model = MAR_Audio(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model