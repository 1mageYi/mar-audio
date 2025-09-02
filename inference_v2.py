# inference_v2.py
import torch
import argparse
import os
from transformers import T5Tokenizer, T5EncoderModel

# 确保你的模型和VAE wrapper的导入路径是正确的
from models.mar_audio import mar_audio_large # 使用V2模型
from models.vae_audio import VAEWrapper
import util.misc as misc

def get_args_parser():
    parser = argparse.ArgumentParser('V2 Inference Script', add_help=False)
    # --- 模型和路径 ---
    parser.add_argument('--model_name', default='mar_audio_large', type=str, help='Name of the model architecture')
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--vae_config_path', default="configs/stable_audio_2_0_vae.json", type=str)
    parser.add_argument('--vae_ckpt_path', default="pretrained_models/vae/stable_audio_open_vae_weights.pth", type=str)
    parser.add_argument('--tokenizer_path', default='t5-base', type=str)
    
    # --- 生成参数 ---
    parser.add_argument('--prompt', required=True, type=str, help='Text prompt for audio generation')
    parser.add_argument('--cfg_scale', default=7.0, type=float, help='Classifier-Free Guidance scale')
    parser.add_argument('--num_iter', default=100, type=int, help='Number of generation steps')
    parser.add_argument('--temperature', default=1.0, type=float, help='Sampling temperature')
    parser.add_argument('--diffloss_d', type=int, default=12, help='Depth of the Diffusion Loss MLP')
    parser.add_argument('--diffloss_w', type=int, default=1536, help='Width of the Diffusion Loss MLP')
    #】
    # --- 输出和设备 ---
    parser.add_argument('--output_path', default='generated_audio_v2.wav', type=str, help='Path to save the generated audio')
    parser.add_argument('--device', default='cuda', type=str)
    
    return parser

@torch.no_grad()
def main(args):
    print("--- V2 Model Inference ---")
    device = torch.device(args.device)

    # 1. 加载所有需要的模型
    print("加载模型...")
    vae = VAEWrapper(args.vae_config_path, args.vae_ckpt_path, device=device)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    text_encoder = T5EncoderModel.from_pretrained(args.tokenizer_path).to(device).eval()

    # 加载你的MAR模型
    model = mar_audio_large(
        text_feature_dim=text_encoder.config.d_model,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w
    ).to(device).eval()

    # 从检查点加载权重
    print(f"从检查点加载权重: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    
    # 我们使用EMA权重进行推理，因为它们通常效果更好
    ema_state_dict = checkpoint['model_ema']
    model.load_state_dict(ema_state_dict)

    # 2. 准备输入
    print(f"准备Prompt: '{args.prompt}'")
    tokens = tokenizer(args.prompt, padding='max_length', truncation=True, max_length=128, return_tensors='pt').input_ids.to(device)
    text_features = text_encoder(tokens).last_hidden_state.mean(dim=1) # V2使用平均池化

    # 3. 执行生成
    print(f"开始生成... (CFG Scale: {args.cfg_scale}, Steps: {args.num_iter})")
    sampled_latents = model.sample_tokens(
        text_features=text_features,
        num_iter=args.num_iter,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        progress=True
    )
    
    # 4. 解码并保存
    print(f"解码并保存至: {args.output_path}")
    vae.decode_batch(sampled_latents, [args.output_path])
    
    print("\n🎉 生成完成！🎉")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)