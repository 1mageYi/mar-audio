import torch
import torchaudio
import json
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.utils import prepare_audio



class VAEWrapper:
    """
    一个用于 Stable Audio VAE 模型的封装类，负责加载模型、
    编码音频文件为潜在向量（latents），以及从潜在向量解码回音频。
    """

    def __init__(self, model_config_path, model_ckpt_path, device=None):
        """
        初始化并加载 VAE 模型。

        参数:
            model_config_path (str): VAE 模型配置 .json 文件的路径。
            model_ckpt_path (str): 预训练的 VAE 权重 .pth 或 .ckpt 文件的路径。
            device (torch.device, 可选): 指定运行模型的设备 (例如 'cuda', 'cpu')。
                                         如果为 None，则自动检测 GPU。
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
        self.target_channels = self.model_config.get("audio_channels", 2)
        self.chunk_size = self.model_config["sample_size"]

        # --- 创建并加载模型 ---
        print("正在创建并加载 VAE 模型...")
        self.model = create_model_from_config(self.model_config)
        
        state_dict = torch.load(model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device).eval()
        print("VAE 模型加载完成。")

    def preprocess_audio(self, audio_path, target_length=441000):
        """
        加载并预处理音频文件以匹配模型输入要求。

        参数:
            audio_path (str): 输入音频文件的路径。
        
        返回:
            torch.Tensor: 预处理后的音频张量，形状为 [1, channels, samples]。
        """
        audio, sample_rate = torchaudio.load(audio_path)

        # 重采样
        audio = prepare_audio(audio,
                              in_sr=sample_rate,
                              target_sr=self.target_sample_rate,
                              target_length=target_length)
        
        # 移动到设备并增加 batch 维度
        return audio.to(self.device).unsqueeze(0)

    def encode(self, input_audio_path):
        """
        将音频文件编码为潜在向量。

        参数:
            input_audio_path (str): 输入音频文件的路径。

        返回:
            torch.Tensor: 编码后的潜在向量张量。
        """
        print(f"正在对 '{input_audio_path}' 进行编码...")
        audio_tensor = self.preprocess_audio(input_audio_path)
        
        print(f"预处理后音频形状: {audio_tensor.shape}")
        
        with torch.no_grad():
            latents = self.model.encode_audio(
                audio_tensor, 
                chunk_size=self.chunk_size, 
            )
        print("编码完成。")
        return latents

    def decode(self, latents, output_audio_path):
        """
        将潜在向量解码为音频文件。

        参数:
            latents (torch.Tensor): 从 encode 方法得到的潜在向量张量。
            output_audio_path (str): 保存解码后音频的路径。
        """
        print(f"正在解码潜在向量...")
        with torch.no_grad():
            reconstructed_audio = self.model.decode_audio(
                latents, 
                chunk_size=self.chunk_size, 
            )
        
        # 移除 batch 维度并保存文件
        reconstructed_audio = reconstructed_audio.squeeze(0)
        torchaudio.save(output_audio_path, reconstructed_audio.cpu(), self.target_sample_rate)
        print(f"解码完成，音频已保存至 '{output_audio_path}'")


def main():
    """
    主函数，用于演示如何使用 VAEWrapper 类。
    """
    # --- 请修改这里的路径 ---
    model_config_path = "path/to/your/stable_audio_1_0_vae.json"
    model_ckpt_path = "path/to/your/stable_audio_open_vae_weights.pth"
    input_audio = "path/to/your/input_audio.wav"
    output_audio = "path/to/your/reconstructed_audio.wav"

    try:
        # 1. 初始化封装类
        vae = VAEWrapper(model_config_path, model_ckpt_path)
        
        # 2. 编码音频文件
        latents = vae.encode(input_audio)
        
        # 打印一下 latent 的形状，以供参考
        print(f"得到的 Latent 形状: {latents.shape}")

        # 3. 从 latent 解码回音频
        vae.decode(latents, output_audio)
        
        print("\n🎉 音频重建流程成功完成！🎉")

    except FileNotFoundError:
        print("\n❌ 错误: 找不到文件。请确保你已经正确设置了 `model_config_path`, `model_ckpt_path` 和 `input_audio` 的路径。")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")


if __name__ == "__main__":
    main()