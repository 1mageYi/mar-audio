# mar_audio_main.py
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from dataset.dataset_audio import AudioTextDataset # <-- 导入新的数据集类
from models.vae_audio import VAEWrapper # <-- 导入你的 VAE Wrapper
from models.mar_audio import mar_audio_large # <-- 导入新的音频MAR模型
from engine_mar_audio import train_one_epoch, evaluate # <-- 导入新的引擎
import copy

def get_args_parser():
    parser = argparse.ArgumentParser('MAR-Audio training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus)')
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_audio_large', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--max_seq_len', default=1024, type=int, help='Max sequence length of audio latents')
    parser.add_argument('--audio_embed_dim', default=64, type=int, help='Audio VAE latent embedding dimension')

    # VAE parameters
    parser.add_argument('--vae_config_path', default="configs/stable_audio_2_0_vae.json", type=str, help='Path to VAE config')
    parser.add_argument('--vae_ckpt_path', default="pretrained_models/vae/stable_audio_open_vae_weights.pth", type=str, help='Path to VAE checkpoint') #

    # Text Tokenizer parameters
    parser.add_argument('--tokenizer_path', default='t5-base', type=str, help='Path or name of T5 tokenizer')
    parser.add_argument('--max_text_len', default=128, type=int, help='Max length for text tokens')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int, help='Number of autoregressive iterations for generation')
    parser.add_argument('--num_eval_samples', default=10, type=int, help='Number of samples to generate for evaluation')
    parser.add_argument('--cfg_scale', default=4.0, type=float, help="Classifier-free guidance scale")
    parser.add_argument('--uncond_prob', default=0.1, type=float, help='Probability of unconditional training for CFG')
    parser.add_argument('--eval_freq', type=int, default=20, help='Evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='Save last checkpoint frequency')
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_bsz', type=int, default=4, help='Generation batch size for evaluation')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay (default: 0.02)')
    parser.add_argument('--grad_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='Epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # Diffusion & MAR params (can be kept as in the original)
    parser.add_argument('--mask_ratio_min', type=float, default=0.5, help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0, help='Gradient clip')
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--temperature', default=1.0, type=float, help='Diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./dummy_data.txt', type=str,
                        help='Path to a text file containing (audio_path, text) pairs, one per line, separated by a tab.')

    parser.add_argument('--output_dir', default='./output_dir_audio',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='Url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # === Setup logging ===
    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # === Setup VAE, Tokenizer ===
    print("Initializing VAE, Tokenizer...")
    vae = VAEWrapper(model_config_path=args.vae_config_path, model_ckpt_path=args.vae_ckpt_path, device=device) #
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path, model_max_length=args.max_text_len)

    # === Setup Dataset ===
    # This is a placeholder for your actual data loading.
    # Create a text file `dummy_data.txt` with lines like:
    # path/to/audio1.wav	a dog is barking
    # path/to/audio2.wav	techno music with a heavy bassline
    print("Loading data...")
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            audio_text_pairs = [line.strip().split('\t') for line in f]
    except FileNotFoundError:
        print(f"Warning: Data file not found at {args.data_path}. Using placeholder data.")
        print("Please create a tab-separated file with audio paths and text prompts.")
        # Create a dummy file for demonstration
        with open(args.data_path, 'w') as f:
            f.write("placeholder.wav\tthis is a test prompt\n")
        if not os.path.exists("placeholder.wav"):
            # Create a silent wav file
            import soundfile as sf
            sf.write("placeholder.wav", np.zeros(44100), 44100)
        audio_text_pairs = [("placeholder.wav", "this is a test prompt")]


    dataset_train = AudioTextDataset(data=audio_text_pairs, tokenizer=tokenizer, max_text_len=args.max_text_len)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # === Setup Model ===
    print(f"Creating model: {args.model}")
    model = mar_audio_large(
        max_seq_len=args.max_seq_len,
        audio_embed_dim=args.audio_embed_dim,
        vocab_size=tokenizer.vocab_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
    )
    model.to(device)
    model_without_ddp = model

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    # === Setup Optimizer ===
    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Effective batch size: {eff_batch_size}")
    print(f"Learning rate: {args.lr:.2e}")

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # === Resume from checkpoint ===
    model_params = list(model_without_ddp.parameters())
    ema_params = copy.deepcopy(model_params)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, ema_params=ema_params)

    if args.evaluate:
        print("Running evaluation only.")
        evaluate(model_without_ddp, vae, tokenizer, args, 0, log_writer=log_writer)
        return

    # === Training Loop ===
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, ema_params=ema_params, epoch_name="last"
            )

        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, tokenizer, args, epoch, log_writer=log_writer)
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)