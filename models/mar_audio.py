# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Block


# --- 我们在这里直接定义这个函数，方便使用 ---
def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    生成一维正弦位置编码
    :param embed_dim: an int, a power of 2.
    :param length: an int.
    :param cls_token: bool, whether to contain CLS token.
    :return: a numpy array of shape [length, embed_dim] or [1+length, embed_dim]
    """
    import numpy as np
    
    assert embed_dim % 2 == 0
    
    position = np.arange(length, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * -(np.log(10000.0) / embed_dim))
    
    pos_embed = np.zeros((length, embed_dim), dtype=np.float32)
    pos_embed[:, 0::2] = np.sin(position * div_term)
    pos_embed[:, 1::2] = np.cos(position * div_term)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        
    return pos_embed


class MAR(nn.Module):
    """
    Masked Auto-Regressive Model for Audio Generation

    核心修改:
    - 移除了图像特有的 grid_size, patch_size, in_chans 等。
    - 引入了音频特有的 max_seq_len (最大序列长度) 和 latent_dim (VAE潜空间维度)。
    - 将 patch_embed 从 Conv2d 改为 Linear，处理 [B, D, L] -> [B, L, D] 格式的音频潜空间。
    - 将位置编码从二维改为一维。
    """
    def __init__(self, 
                 max_seq_len=1024, 
                 latent_dim=64,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 grad_checkpointing=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.grad_checkpointing = grad_checkpointing

        # 1. 嵌入层 (Embedding Layer)
        # 将每个时间步的 latent_dim 维向量映射到 Transformer 的 embed_dim 维
        self.patch_embed = nn.Linear(latent_dim, embed_dim)

        # 2. 位置编码 (Positional Encoding)
        # 创建一维的位置编码，长度为 VAE 支持的最大长度
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim), requires_grad=False)
        pos_embed_val = get_1d_sincos_pos_embed(embed_dim, max_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_val).float().unsqueeze(0))

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 4. 输出头 (Output Head)
        # 将 Transformer 的输出映射回音频 VAE 的潜空间维度
        self.head = nn.Linear(embed_dim, latent_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化所有线性层和层归一化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用 xavier_uniform 初始化
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, z, cond, mask):
        """
        z: 音频 VAE 的潜空间, shape: [B, D, L], e.g., [B, 64, 215]
        cond: 条件 (例如类别嵌入), shape: [B, 1, C]
        mask: 自回归掩码, shape: [B, L, L]
        """
        # (B, D, L) -> (B, L, D)
        z = z.permute(0, 2, 1)
        
        # 嵌入 latent
        # (B, L, D) -> (B, L, C)
        z = self.patch_embed(z)

        # 添加位置编码
        # z 的序列长度 L 可能小于 max_seq_len
        seq_len = z.shape[1]
        z = z + self.pos_embed[:, :seq_len, :]

        # 拼接条件 (如果提供)
        if cond is not None:
            x = torch.cat((cond, z), dim=1)
        else:
            x = z

        # 应用 Transformer Blocks
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        # 移除条件 token
        if cond is not None:
            x = x[:, cond.shape[1]:, :]
            
        return x

    def forward(self, z, cond=None, mask=None):
        """
        z: 音频 VAE 的潜空间, shape: [B, D, L]
        cond: 条件, e.g., class token
        mask: 自回归掩码
        """
        # 通过 Transformer 编码器获取隐藏状态
        latent = self.forward_encoder(z, cond, mask) # [B, L, C]
        
        # 通过输出头预测下一个 latent vector
        pred = self.head(latent) # [B, L, D]
        
        # 维度转换以匹配 VAE 输入格式
        # (B, L, D) -> (B, D, L)
        pred = pred.permute(0, 2, 1)
        
        return pred


# ----------------------------------------------------------------------------
# 模型配置的工厂函数
# ----------------------------------------------------------------------------

def mar_base(**kwargs):
    model = MAR(
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mar_large(**kwargs):
    model = MAR(
        embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mar_huge(**kwargs):
    model = MAR(
        embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# 将字符串映射到模型构造函数
mar_models = {
    'mar_base': mar_base,
    'mar_large': mar_large,
    'mar_huge': mar_huge,
}

if __name__ == '__main__':
    # 一个简单的测试来验证模型的维度流动
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟音频 VAE 的输出
    # 10s 音频, VAE latent 形状为 [B, D, L]
    dummy_latent = torch.randn(2, 64, 215).to(device) # Batch=2, Dim=64, Length=215

    # 实例化一个基础版本的 MAR 模型
    # 使用你的音频 VAE 参数
    model = mar_base(
        max_seq_len=1024,
        latent_dim=64
    ).to(device)
    
    print("模型实例化成功，放置在:", device)
    print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")

    # 前向传播测试
    with torch.no_grad():
        output_latent = model(dummy_latent)
    
    print("\n--- 维度测试 ---")
    print("输入 Latent 形状:", dummy_latent.shape)
    print("输出 Latent 形状:", output_latent.shape)
    
    assert dummy_latent.shape == output_latent.shape, "输入和输出形状不匹配！"
    print("\n维度测试通过！代码已准备好用于音频生成任务。")