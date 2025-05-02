import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
from config.config import *

class ProteinDataset(Dataset):
    """蛋白质序列数据集"""
    
    def __init__(
        self,
        sequences: Optional[List[str]] = None,
        split: str = "train",
        max_length: int = 1000
    ):
        """初始化数据集
        
        Args:
            sequences: 序列列表，如果为None则从文件加载
            split: 数据集划分（"train"或"val"）
            max_length: 最大序列长度
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        
        if sequences is None:
            # 从文件加载序列
            data_dir = Path("data/raw")
            if split == "train":
                file_path = data_dir / "train_sequences.txt"
            else:
                file_path = data_dir / "val_sequences.txt"
            
            with open(file_path, "r") as f:
                sequences = [line.strip() for line in f if line.strip()]
        
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # 编码序列
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 获取input_ids
        input_ids = encoded["input_ids"].squeeze(0)
        
        # 获取embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "embeddings": embeddings
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批处理函数
        
        Args:
            batch: 批数据
            
        Returns:
            批处理后的数据
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        embeddings = torch.stack([item["embeddings"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "embeddings": embeddings
        }

def create_data_loaders(
    embeddings: torch.Tensor,
    batch_size: int = VAE_BATCH_SIZE,
    train_split: float = TRAIN_TEST_SPLIT,
    num_workers: int = 4,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    dataset = ProteinDataset(embeddings)
    train_size = int((1 - train_split) * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader 