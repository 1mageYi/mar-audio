from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block
from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    """根据给定的顺序和长度创建掩码。"""
    masking = torch.zeros(bsz, seq_len, device=order.device)
    # 确保 mask_len 是一个标量整数
    mask_len_scalar = mask_len.long().item() if mask_len.numel() == 1 else mask_len.long()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len_scalar], src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class MAR_Audio(nn.Module):
    """
    适用于音频生成的 MAR (Masked Auto-Regressive) 模型。
    - 使用音频 VAE 的潜码作为输入/输出。
    - 使用文本作为条件。
    """
    def __init__(self,
                 # 音频潜码参数
                 max_seq_len=1024,          # VAE潜码的最大序列长度
                 audio_embed_dim=64,         # VAE潜码的嵌入维度
                 
                 # 文本条件参数
                 vocab_size=10000,           # 假设的词汇表大小
                 text_embed_dim=768,         # 文本嵌入维度
                 
                 # Transformer 架构参数 (与原始MAR保持一致)
                 encoder_embed_dim=1024,
                 encoder_depth=16,
                 encoder_num_heads=16,
                 decoder_embed_dim=1024,
                 decoder_depth=16,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 
                 # MAR 特定参数
                 mask_ratio_min=0.5,         # 音频任务的掩码率可能需要调整
                 buffer_size=64,             # 保持不变
                 
                 # 其余参数
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False):
        super().__init__()

        # --- 音频 & 文本参数 ---
        self.max_seq_len = max_seq_len
        self.audio_embed_dim = audio_embed_dim
        self.grad_checkpointing = grad_checkpointing

        # --- 文本条件嵌入 ---
        # 1. 词嵌入层
        self.text_token_emb = nn.Embedding(vocab_size, text_embed_dim)
        # 2. 一个简单的文本编码器（例如，几个Transformer块）来获得全局文本表征
        #    为了简化，我们这里使用一个简单的线性层将词嵌入的平均值映射到encoder维度
        #    在实际应用中，这里应该是一个更复杂的文本模型（如BERT或自定义Transformer）
        self.text_feature_proj = nn.Linear(text_embed_dim, encoder_embed_dim)
        # 3. 用于无条件生成的“空”文本表征 (类似于CFG中的做法)
        self.null_text_feature = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --- MAR 掩码率生成器 ---
        # 对于音频，我们可以从一个较高的掩码率开始
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --- MAR 编码器 ---
        # 将音频潜码从 audio_embed_dim 投影到 encoder_embed_dim
        self.z_proj = nn.Linear(self.audio_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        
        self.buffer_size = buffer_size
        # 位置编码现在需要能够处理最大序列长度
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.max_seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --- MAR 解码器 ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.max_seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.max_seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --- Diffusion Loss (与原来完全一样) ---
        self.diffloss = DiffLoss(
            target_channels=self.audio_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        torch.nn.init.normal_(self.text_token_emb.weight, std=.02)
        torch.nn.init.normal_(self.null_text_feature, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None: nn.init.constant_(m.bias, 0)
            if m.weight is not None: nn.init.constant_(m.weight, 1.0)
    
    def encode_text(self, text_tokens):
        """将文本 token ID 编码为条件特征。"""
        # [B, T] -> [B, T, D_text]
        text_embeds = self.text_token_emb(text_tokens) 
        # [B, T, D_text] -> [B, D_text] (简单平均池化)
        text_features = text_embeds.mean(dim=1) 
        # [B, D_text] -> [B, D_encoder]
        return self.text_feature_proj(text_features)

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

    def forward_mae_encoder(self, x, mask, text_feature):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
        
        # 将文本特征放入 buffer
        x[:, :self.buffer_size] = text_feature.unsqueeze(1)

        # 加上位置编码 (注意，只取当前序列长度所需的部分)
        x = x + self.encoder_pos_embed_learned[:, :self.buffer_size + seq_len, :]
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # Transformer blocks
        for block in self.encoder_blocks:
            x = checkpoint(block, x) if self.grad_checkpointing else block(x)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(self, x, mask):
        x = self.decoder_embed(x)
        seq_len = mask.shape[1]
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(-1, x.shape[-1])
        
        x = x_after_pad + self.decoder_pos_embed_learned[:, :self.buffer_size + seq_len, :]

        for block in self.decoder_blocks:
            x = checkpoint(block, x) if self.grad_checkpointing else block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned[:, :seq_len, :]
        return x

    def forward_loss(self, z, target, mask):
        # 这个函数基本不需要改动
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, audio_latents, text_tokens, uncond_prob=0.1):
        """
        模型前向传播。
        :param audio_latents: VAE编码后的音频潜码, [B, L, D_audio]。
        :param text_tokens: 分词后的文本ID, [B, T]。
        :param uncond_prob: 无条件训练的概率 (用于CFG)。
        """
        bsz, seq_len, _ = audio_latents.shape
        device = audio_latents.device
        
        # 1. 获取文本条件
        text_features = self.encode_text(text_tokens)

        # 2. Classifier-Free Guidance (CFG) 训练
        #    以一定概率将文本条件替换为 null embedding
        if self.training:
            uncond_mask = torch.rand(bsz, 1, device=device) < uncond_prob
            text_features = torch.where(uncond_mask, self.null_text_feature, text_features)

        # 3. patchify -> 在这里我们不需要 patchify，直接使用 latents
        x = audio_latents
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz, seq_len, device)
        mask = self.random_masking(x, orders)

        # 4. MAE Encoder
        x = self.forward_mae_encoder(x, mask, text_features)

        # 5. MAE Decoder
        z = self.forward_mae_decoder(x, mask)

        # 6. Diffusion Loss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    @torch.no_grad()
    def sample_tokens(self, text_tokens, num_iter=64, cfg_scale=4.0, temperature=1.0, progress=True):
        """
        从文本生成音频潜码。
        :param text_tokens: 条件文本的token ID [B, T]。
        :param num_iter: 迭代生成步数。
        :param cfg_scale: CFG尺度。
        :param temperature: 采样温度。
        """
        bsz, _ = text_tokens.shape
        device = text_tokens.device
        
        # 确定生成长度，这里我们先固定一个长度，例如 215
        # 在实际应用中，可以由文本或其他方式决定
        seq_len = 215

        # 1. 编码文本条件
        text_features = self.encode_text(text_tokens)

        # 2. 初始化全掩码的序列和随机顺序
        mask = torch.ones(bsz, seq_len, device=device)
        tokens = torch.zeros(bsz, seq_len, self.audio_embed_dim, device=device)
        orders = self.sample_orders(bsz, seq_len, device)

        indices = tqdm(range(num_iter)) if progress else range(num_iter)

        for step in indices:
            # 准备CFG的输入
            model_in_tokens = torch.cat([tokens, tokens], dim=0)
            model_in_mask = torch.cat([mask, mask], dim=0)
            cond_features = torch.cat([text_features, self.null_text_feature.repeat(bsz, 1)], dim=0)

            # Encoder -> Decoder
            z_cond, z_uncond = self.forward_mae_decoder(
                self.forward_mae_encoder(model_in_tokens, model_in_mask, cond_features),
                model_in_mask
            ).chunk(2)
            
            # 执行CFG
            z = z_uncond + cfg_scale * (z_cond - z_uncond)

            # 确定本轮需要预测的token
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor(np.floor(seq_len * mask_ratio), device=device)
            mask_len = torch.max(torch.tensor(1.0, device=device), torch.min(mask.sum(dim=-1) - 1, mask_len))

            mask_next = mask_by_order(mask_len, orders, bsz, seq_len)
            mask_to_pred = (mask.bool() ^ mask_next.bool())
            
            # 从 Diffusion Head 采样
            z_pred = z[mask_to_pred.nonzero(as_tuple=True)]
            
            sampled_latents = self.diffloss.sample(z_pred, temperature=temperature, cfg=1.0) # 此处的cfg是diffloss内部的，设为1.0
            
            # 将新采样的token放回序列
            tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_latents
            
            # 更新掩码
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