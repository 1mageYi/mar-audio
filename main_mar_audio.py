# main_mar.py

import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio

# 导入你的模型和数据加载器
from models.mar import mar_models
from models.diffloss import DiffLoss
from models.audio_vae import AudioVAE
from data.audio_dataset import AudioDataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAR for Audio Generation', add_help=False)

    # -- 模型参数 --
    parser.add_argument('--model', default='mar_base', type=str, metavar='MODEL',
                        help=f'Name of model to train: {", ".join(mar_models.keys())}')
    parser.add_argument('--max_seq_len', default=1024, type=int, help='Max sequence length of VAE latent')
    parser.add_argument('--latent_dim', default=64, type=int, help='Dimension of VAE latent space')
    parser.add_argument('--audio_vae_path', type=str, required=True, help='Path to your pretrained audio VAE model')

    # -- DiffLoss 参数 --
    parser.add_argument('--diffloss_d', default=3, type=int, help='Depth of DiffLoss MLP')
    parser.add_argument('--diffloss_w', default=1024, type=int, help='Width of DiffLoss MLP')

    # -- 训练参数 --
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio for training')
    
    # -- 数据集参数 --
    parser.add_argument('--data_path', type=str, required=True, help='path to your audio dataset')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--duration', type=int, default=10, help='Audio duration in seconds')
    
    # -- 其他 --
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save checkpoints')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate during evaluation')

    return parser


def main(args):
    # ---- 设置环境 ----
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载模型 ----
    print("Loading models...")
    # 1. 加载你的音频 VAE
    vae = AudioVAE(model_path=args.audio_vae_path).to(device)
    vae.eval() # VAE不参与训练

    # 2. 实例化 MAR 模型
    mar_model = mar_models[args.model](
        max_seq_len=args.max_seq_len,
        latent_dim=args.latent_dim,
    ).to(device)

    # 3. 实例化 DiffLoss
    loss_func = DiffLoss(
        target_channels=args.latent_dim,
        z_channels=mar_model.embed_dim, # DiffLoss以MAR的隐藏状态为条件
        depth=args.diffloss_d,
        width=args.diffloss_w,
        num_sampling_steps=256 # 默认值，可以调整
    ).to(device)

    # 将 MAR 和 DiffLoss 的参数合并给优化器
    optimizer = torch.optim.AdamW(
        list(mar_model.parameters()) + list(loss_func.parameters()),
        lr=args.lr
    )
    
    print(f"MAR Model: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in mar_model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"DiffLoss parameters: {sum(p.numel() for p in loss_func.parameters() if p.requires_grad)/1e6:.2f}M")


    # ---- 加载数据 ----
    print("Loading dataset...")
    dataset = AudioDataset(
        root_dir=args.data_path,
        max_duration_secs=args.duration,
        sample_rate=args.sample_rate
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # ---- 如果是评估模式 ----
    if args.evaluate:
        print("Running in evaluation mode...")
        generate_samples(mar_model, vae, args.num_samples, device, args)
        return

    # ---- 训练循环 ----
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        mar_model.train()
        loss_func.train()
        
        total_loss = 0
        for step, audio_wave in enumerate(data_loader):
            audio_wave = audio_wave.to(device)

            # 1. 使用 VAE 获取潜空间表示
            with torch.no_grad():
                z = vae.encode(audio_wave) # Shape: [B, D, L]

            # 2. *** 这是 MAR 训练的核心：随机掩码 ***
            B, D, L = z.shape
            
            # 生成一个随机的序列顺序
            noise = torch.rand(B, L, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # 决定要保留多少作为上下文（不被掩码）
            len_keep = int(L * (1 - args.mask_ratio))
            
            # 得到上下文的索引和被掩码的索引
            ids_keep = ids_shuffle[:, :len_keep]
            ids_mask = ids_shuffle[:, len_keep:]
            
            # 从原始潜变量 z 中，根据索引选出上下文部分
            # 首先调整 z 的维度以方便索引: [B, D, L] -> [B, L, D]
            z_permuted = z.permute(0, 2, 1)
            z_context = torch.gather(z_permuted, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

            # 3. 通过 MAR 模型处理上下文
            # 注意：这里没有传入 mask 参数，因为上下文内部是双向可见的！
            mar_output = mar_model.forward_encoder(z_context.permute(0, 2, 1), cond=None, mask=None)

            # 4. 计算 DiffLoss
            # 目标是让模型根据上下文预测出被掩码的部分
            # 收集被掩码的目标潜变量 (ground truth)
            target_masked = torch.gather(z_permuted, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
            
            # 收集对应的 MAR 模型输出
            pred_masked = torch.gather(mar_output, dim=1, index(ids_mask.unsqueeze(-1).expand(-1, -1, mar_model.embed_dim)))

            # 计算损失
            loss = loss_func(target=target_masked, z=pred_masked)
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"--- Epoch {epoch} finished. Average Loss: {avg_loss:.4f} ---")
        
        # 保存 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch+1}.pth')
            torch.save({
                'mar_model': mar_model.state_dict(),
                'diffloss': loss_func.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


    total_time = time.time() - start_time
    print(f"Training finished in {total_time/3600:.2f} hours")


@torch.no_grad()
def generate_samples(mar_model, vae, num_samples, device, args):
    """
    使用自回归方式生成样本
    """
    mar_model.eval()
    
    # 准备一个随机的起始序列（这里用零初始化）
    B, D, L = num_samples, args.latent_dim, args.max_seq_len
    z_generated = torch.zeros(B, D, L, device=device)
    
    # 生成随机的生成顺序
    ids_shuffle = torch.argsort(torch.rand(B, L, device=device), dim=1)
    
    print("Starting autoregressive generation...")
    for step in range(L):
        # 当前要生成的 token 的索引
        current_ids = ids_shuffle[:, step:step+1]
        
        # 已经生成的 token 作为上下文
        context_ids = ids_shuffle[:, :step]
        
        if step == 0:
            # 第一步没有上下文，可以传入一个可学习的起始 token 或零向量
            context_mar_input = torch.zeros(B, D, 0, device=device)
        else:
            # 从已生成的 z_generated 中收集上下文
            context_z = torch.gather(z_generated.permute(0,2,1), dim=1, index=context_ids.unsqueeze(-1).expand(-1,-1,D))
            context_mar_input = context_z.permute(0,2,1)

        # 使用 MAR 模型处理上下文
        mar_output = mar_model.forward_encoder(context_mar_input, cond=None, mask=None)

        # 采样 (这里为了简单直接取模型输出，实际应使用 DiffLoss 的 sample 方法)
        # 假设要预测的位置的 MAR 输出为
        if step == 0:
            # 假设第一个输出对应第一个预测
            pred_hidden_state = mar_output[:, 0, :]
        else:
            # 在MAR输出中找到对应当前要生成位置的输出
            # 这部分逻辑较为复杂，简化处理：总是取最后一个输出
            pred_hidden_state = mar_output[:, -1, :]
            
        # *** 此处应调用 DiffLoss.sample() 来获得更真实的采样 ***
        # 为了简化，我们直接用 MAR 的 head 来做一个粗略的预测
        pred_z_value = mar_model.head(pred_hidden_state) # Shape: [B, D]

        # 将预测出的值填充到 z_generated 的对应位置
        # scatter_ 方法可以在指定索引上填充值
        z_generated.permute(0,2,1).scatter_(dim=1, index=current_ids.unsqueeze(-1).expand(-1,-1,D), src=pred_z_value.unsqueeze(1))
        
        if (step + 1) % 100 == 0:
            print(f"Generated {step+1}/{L} tokens...")
            
    print("Generation finished. Decoding with VAE...")
    # 使用 VAE 解码
    generated_audio = vae.decode(z_generated)
    
    # 保存音频
    save_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_samples):
        file_path = os.path.join(save_dir, f"generated_sample_{i}.wav")
        torchaudio.save(file_path, generated_audio[i].cpu(), args.sample_rate)
        print(f"Saved to {file_path}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)