import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class AudioTextDataset(Dataset):
    """
    一个用于加载（音频路径，文本描述）数据对的数据集。
    这个版本会从一个CSV文件和音频目录中加载数据。

    参数:
        csv_path (str): CSV文件的路径。
                     CSV文件应包含 'youtube_id', 'start_time', 和 'caption' 列。
        audio_dir (str): 存放所有 .wav 音频文件的目录路径。
        tokenizer: 从 transformers 加载的预训练分词器 (例如 T5Tokenizer)。
        max_text_len (int): 文本分词后的最大长度。
    """
    def __init__(self, csv_path, audio_dir, tokenizer, max_text_len=128):
        super().__init__()
        
        # 1. 加载和验证输入
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")
        if not os.path.isdir(audio_dir):
            raise NotADirectoryError(f"音频目录未找到: {audio_dir}")
            
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

        # 2. 使用 pandas 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 3. 准备数据列表
        # self.data 是一个列表，每个元素是一个元组 (audio_path, caption)
        self.data = []
        for index, row in df.iterrows():
            # 从 'youtube_id' 和 'start_time' 构建文件名
            # print(row['audiocap_id'])
            filename = f"{row['youtube_id']}_{int(row['start_time'])}.wav"
            

            # 构建完整的音频文件路径
            audio_path = os.path.join(self.audio_dir, filename)
            
            # 获取文本描述
            caption = row['caption']
            
            self.data.append((audio_path, caption))

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.data)

    def __getitem__(self, index):
        """
        获取数据集中的一个样本。

        参数:
            index (int): 样本的索引。

        返回:
            tuple: (audio_path, token_ids)
                - audio_path (str): 音频文件的完整路径。
                - token_ids (torch.Tensor): 经过分词和编码后的文本ID。
        """
        # 从 self.data 中获取音频路径和原始文本
        audio_path, text = self.data[index]

        # 使用分词器处理文本
        # 这会进行分词、填充到max_length、截断并转换为PyTorch张量
        tokenized_output = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        
        # 获取 token IDs 并移除不必要的 batch 维度（从 [1, 128] 变为 [128]）
        token_ids = tokenized_output.input_ids.squeeze(0)

        return audio_path, token_ids
    

if __name__ == "__main__":
    # 示例用法
    from transformers import T5Tokenizer

    # 假设 CSV 文件和音频目录已存在
    csv_path = "/root/data1/yimingjing/data/audiocaps/dataset2.0/train.csv"
    audio_dir = "/root/data1/yimingjing/data/audiocaps/audiocaps_raw_audio"

    # 加载分词器
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # 创建数据集实例
    dataset = AudioTextDataset(csv_path, audio_dir, tokenizer)

    # 打印数据集大小和第一个样本
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])