# dataset_audio.py
import torch
from torch.utils.data import Dataset

class AudioTextDataset(Dataset):
    """
    一个用于加载（音频路径，文本描述）数据对的数据集。

    参数:
        data (list): 一个列表，每个元素是一个元组 (audio_path, text_prompt)。
                     例如: [("path/to/audio1.wav", "a dog barking"), ("path/to/audio2.wav", "techno music")]
        tokenizer: 从 transformers 加载的预训练分词器 (例如 T5Tokenizer)。
        max_text_len (int): 文本分词后的最大长度。
    """
    def __init__(self, data, tokenizer, max_text_len=128):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_path, text = self.data[index]

        # 使用分词器处理文本
        tokenized_output = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        
        # 获取 token IDs 并移除不必要的 batch 维度
        token_ids = tokenized_output.input_ids.squeeze(0)

        # 返回音频路径和编码后的文本ID
        # VAE编码将在主训练循环中完成，这里只返回路径
        return audio_path, token_ids