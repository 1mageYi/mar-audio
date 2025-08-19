export http_proxy=http://10.1.0.10:6899
export https_proxy=http://10.1.0.10:6899


export CUDA_VISIBLE_DEVICES=5,6


torchrun --nproc_per_node=2 main_mar_audio.py \
    --model mar_audio_large \
    --metadata_path "/root/data1/yimingjing/data/audiocaps/dataset2.0" \
    --raw_data_path "/root/data1/yimingjing/data/audiocaps/audiocaps_raw_audio" \
    --vae_config_path "configs/stable_audio_2_0_vae.json" \
    --vae_ckpt_path "pretrained_models/vae/stable_audio_open_vae_weights.pth" \
    --output_dir "./output_audiocaps_training_multi_gpu" \
    --batch_size 8 \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1e-4 \
    --ema_rate 0.9999 \
    --uncond_prob 0.1 \
    --cfg_scale 4.0 \
    --num_workers 8 \
    --eval_freq 10