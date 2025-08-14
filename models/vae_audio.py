import torch
import torchaudio
import json
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.utils import prepare_audio
import numpy as np


class VAEWrapper:
    """
    一个用于 Stable Audio VAE 模型的封装类，负责加载模型、
    编码音频文件为潜在向量（latents），以及从潜在向量解码回音频。
    (已更新以支持批量处理)
    """

    def __init__(self, model_config_path, model_ckpt_path, target_length=441000, device=None):
        """
        初始化并加载 VAE 模型。
        """
        print("正在初始化 VAE Wrapper...")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")

        # --- 加载模型配置 ---
        with open(model_config_path) as f:
            self.model_config = json.load(f)
        self.target_sample_rate = self.model_config["sample_rate"]
        self.chunk_size = self.model_config["sample_size"]
        self.target_channels = self.model_config.get("audio_channels", 2)
        self.target_length = target_length

        # --- 创建并加载模型 ---
        print("正在创建并加载 VAE 模型...")
        self.model = create_model_from_config(self.model_config)
        
        state_dict = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device).eval()


        print("VAE 模型加载完成。")

    def preprocess_audio(self, audio_path):
        """
        加载并预处理单个音频文件。
        """
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            audio = prepare_audio(audio,
                                  in_sr=sample_rate,
                                  target_sr=self.target_sample_rate,
                                  target_length=self.target_length,
                                  target_channels=self.target_channels,
                                  device=self.device,
                                  )
            return audio.squeeze(0) 
        except Exception as e:
            print(f"处理音频文件 {audio_path} 时出错: {e}. 返回静音。")
            return torch.zeros((self.model_config.get("audio_channels", 2), self.target_length)).to(self.device)

    def encode(self, input_audio_path):
        """
        将【单个】音频文件编码为潜在向量。
        """
        audio_tensor = self.preprocess_audio(input_audio_path).unsqueeze(0)
        with torch.no_grad():
            # self.model.encode_audio 期望一个 batch, 所以我们增加一个维度
            latents = self.model.encode_audio(audio_tensor, chunk_size=self.chunk_size)
        return latents.permute(0, 2, 1) # 返回 [B, L, D] = [1, 215, 64]

    def decode(self, latents, output_audio_path):
        """
        将【单个】潜在向量解码为音频文件。
        """
        # 将 [1, L, D] 转回模型期望的 [1, D, L]
        latents_for_decode = latents.permute(0, 2, 1)
        with torch.no_grad():
            reconstructed_audio = self.model.decode_audio(latents_for_decode, chunk_size=self.chunk_size)
        reconstructed_audio = reconstructed_audio.squeeze(0)
        torchaudio.save(output_audio_path, reconstructed_audio.cpu(), self.target_sample_rate)
        
    # --- 新增的批量处理方法 ---
    def encode_batch(self, audio_paths: list):
        """
        将【一批】音频文件编码为潜在向量。
        """
        # preprocess_audio 返回 2D 张量 [channels, length]
        audio_tensors = [self.preprocess_audio(path) for path in audio_paths]
        # torch.stack 会将一列 [channels, length] 的张量正确堆叠为 [batch, channels, length]
        audio_batch = torch.stack(audio_tensors, dim=0)

        with torch.no_grad():
            latents_batch = self.model.encode_audio(audio_batch, chunk_size=self.chunk_size)
        
        return latents_batch.permute(0, 2, 1)

    def decode_batch(self, latents_batch, output_paths: list):
        """
        将【一批】潜在向量解码为多个音频文件。
        """
        if len(latents_batch) != len(output_paths):
            raise ValueError("潜在向量批次大小必须与输出路径列表长度匹配。")

        latents_for_decode = latents_batch.permute(0, 2, 1)
        with torch.no_grad():
            reconstructed_audio_batch = self.model.decode_audio(latents_for_decode, chunk_size=self.chunk_size)
        
        for i in range(reconstructed_audio_batch.shape[0]):
            audio_sample = reconstructed_audio_batch[i].cpu()
            torchaudio.save(output_paths[i], audio_sample, self.target_sample_rate)

if __name__ == "__main__":
    model_config_path = "configs/stable_audio_2_0_vae.json"
    model_ckpt_path = "pretrained_models/vae/stable_audio_open_vae_weights.pth"
    
    # 假设我们有这些文件
    input_audios = ["/root/data1/yimingjing/workspace/stable-audio-tools/output.wav", "/root/data1/yimingjing/workspace/stable-audio-tools/diffusers_test_10s.wav"]
    output_audios = ["test/reconstructed1.wav", "test/reconstructed2.wav"]
    
    try:
        vae = VAEWrapper(model_config_path, model_ckpt_path)
        
        print("\n--- 测试批量编码 ---")
        latents_batch = vae.encode_batch(input_audios)
        print(f"得到的 Latent 批次形状: {latents_batch.shape}")

        print("\n--- 测试批量解码 ---")
        vae.decode_batch(latents_batch, output_audios)
        print(f"解码完成，音频已保存至 {output_audios}")
        
        print("\n🎉 批量处理流程成功完成！🎉")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")