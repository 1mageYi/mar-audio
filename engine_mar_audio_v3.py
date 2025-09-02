# engine_mar_audio.py
import math
import sys
import os
import copy
import time
from typing import Iterable

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def train_one_epoch(model: torch.nn.Module, vae: torch.nn.Module, text_encoder,
                    model_params: Iterable, ema_params: Iterable,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, (audio_paths, text_tokens) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        text_tokens = text_tokens.to(device, non_blocking=True)
        
        with torch.no_grad():
            audio_latents = vae.encode_batch(audio_paths).to(device)
            # --- 关键：在这里进行文本编码 ---
            # 使用 T5EncoderModel 获取 last_hidden_state，然后取平均
            text_features = text_encoder(input_ids=text_tokens).last_hidden_state
            

        # --- 关键调试点 1: 检查输入数据 ---
        # 我们只在第一个step和主进程打印，避免刷屏
        if data_iter_step == 0 and misc.is_main_process():
            print("\n--- [DEBUG] Input Data Check ---")
            print(f"Audio Latents shape: {audio_latents.shape}")
            print(f"Audio Latents mean/std: {audio_latents.mean():.4f}/{audio_latents.std():.4f}")
            print(f"Text Features shape: {text_features.shape}")
            print(f"Text Features mean/std: {text_features.mean():.4f}/{text_features.std():.4f}")
            print("--------------------------------\n")
        # ------------------------------------


        with torch.cuda.amp.autocast():
            # 将 text_features 传入模型
            loss = model(audio_latents, text_features, uncond_prob=args.uncond_prob)


        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # --- 优化步骤 ---
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)


        # --- 关键调试点 2: 检查梯度流 ---
        # 同样只在第一个step和主进程打印
        if data_iter_step == 0 and misc.is_main_process():
            # 挑选模型中几个关键参数来检查梯度
            # 比如编码器的第一个block的权重和解码器最后一个block的权重
            grad_norm_encoder = model.module.encoder_blocks[0].attn.qkv.weight.grad.norm().item()
            grad_norm_decoder = model.module.decoder_blocks[-1].mlp.fc2.weight.grad.norm().item()
            print("\n--- [DEBUG] Gradient Flow Check ---")
            print(f"Loss value: {loss_value:.4f}")
            print(f"Encoder grad norm: {grad_norm_encoder:.6f}")
            print(f"Decoder grad norm: {grad_norm_decoder:.6f}")
            print("---------------------------------\n")
        # ------------------------------------

        
        optimizer.zero_grad()
        torch.cuda.synchronize()

        # --- 更新 EMA 模型 ---
        update_ema(ema_params, model_params, rate=args.ema_rate)

        # --- 日志记录 ---
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and data_iter_step % print_freq == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)

    # 同步所有进程的统计数据
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model_without_ddp: torch.nn.Module, vae: torch.nn.Module, ema_params: Iterable, tokenizer, text_encoder,
             args, epoch: int, log_writer=None, use_ema=True):
    model_without_ddp.eval()
    if misc.is_main_process():
        # --- 切换到 EMA 参数进行评估 ---
        if use_ema:
            print("Switching to EMA model for evaluation...")
            model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
            # 需要一个函数来从 ema_params 列表构建 state_dict
            # 假设 misc.load_ema_to_model 可以做到这一点
            misc.load_ema_to_model(model_without_ddp, ema_params) # 你需要实现这个函数

        # --- 准备评估用的文本提示 ---
        eval_prompts = [
            "motor noise is followed by a horn honking and a siren wailing",
            "pigeons vocalize and birds chirp",
            "a harsh wind blows as a man speaks and another man speaks",
            "vehicles pass by on a roadway",
            "a toilet flushes and water sputters as it drains",
        ]
        # 限制评估样本数量
        eval_prompts = eval_prompts[:args.num_eval_samples]
        if len(eval_prompts) == 0:
            print("No evaluation prompts provided or num_eval_samples is 0. Skipping evaluation.")
            return

        print(f"Generating {len(eval_prompts)} samples for evaluation...")
        text_tokens = tokenizer(
            eval_prompts,
            padding='max_length',
            truncation=True,
            max_length=args.max_text_len,
            return_tensors='pt'
        ).input_ids.to(args.device)
        
        text_features = text_encoder(input_ids=text_tokens).last_hidden_state
        # text_features = text_features.mean(dim=1)

        # --- 生成音频潜码 ---
        start_time = time.time()
        sampled_latents = model_without_ddp.sample_tokens(
            text_features=text_features,
            num_iter=args.num_iter,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            progress=True
        )

        # --- 关键修改点: 检查模型坍塌 ---
        # 打印生成潜码的统计数据
        mean_val = torch.mean(sampled_latents).item()
        std_val = torch.std(sampled_latents).item()
        print("\n--- Generated Latent Stats (for collapse check) ---")
        print(f"Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")
        print("-----------------------------------------------------\n")
        # ----------------------------------------------------
        
        # --- 解码和保存 (应用缩放因子) ---
        # 假设你的缩放因子是 1.042592
        # 如果你现在不想用，可以暂时设为 1.0
        scaling_factor = 1.042592
        sampled_latents = sampled_latents / scaling_factor

        save_folder = os.path.join(args.output_dir, f"eval_epoch-{epoch}_ema")
        if misc.is_main_process():
            os.makedirs(save_folder, exist_ok=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")

        # --- 解码并保存音频 ---
        

        for i, latent in enumerate(sampled_latents):
            latent = latent.unsqueeze(0)  # Add batch dimension back
            output_path = os.path.join(save_folder, f"sample_{i+1}_epoch{epoch}.wav")
            # VAE 解码
            vae.decode(latent, output_path)

        print(f"Evaluation samples saved to: {save_folder}")

        # --- 音频评估指标 ---
        # 图像的 FID/IS 在此不适用。对于音频，可以使用 Fréchet Audio Distance (FAD),
        # CLAP score, 或其他基于分类器的指标。这些通常需要额外的设置和预训练模型。
        # 这里我们只生成样本供主观评估。
        if log_writer is not None:
            # 假设你有一个函数可以计算 FAD
            # fad_score = calculate_fad(save_folder, your_ground_truth_folder)
            # log_writer.add_scalar('eval/fad', fad_score, epoch)
            pass

        # --- 切换回原始模型参数 ---
        if use_ema:
            print("Switching back from EMA model.")
            model_without_ddp.load_state_dict(model_state_dict)

    model_without_ddp.train() # 切换回训练模式
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()