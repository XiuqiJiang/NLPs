import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import os

# 修复导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import (
    ESM_MODEL_NAME,
    BATCH_SIZE,
    TRAIN_TEST_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    MAX_SEQUENCE_LENGTH,
    DATA_PATH,
    ESM_EMBEDDING_DIM
)

class ProteinDataset(Dataset):
    """蛋白质序列数据集"""
    
    def __init__(
        self,
        sequences: Optional[List[str]] = None,
        split: str = "train",
        data_path: str = DATA_PATH
    ):
        """初始化数据集
        
        Args:
            sequences: 序列列表，如果为None则从文件加载
            split: 数据集划分（"train"或"val"）
            data_path: 数据文件路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        self.model.eval()  # 设置为评估模式
        
        if sequences is None:
            # 从文件加载序列
            with open(data_path, "r") as f:
                sequences = [line.strip() for line in f if line.strip()]
            
            # 划分训练集和验证集
            train_size = int(len(sequences) * (1 - TRAIN_TEST_SPLIT))
            if split == "train":
                sequences = sequences[:train_size]
            else:
                sequences = sequences[train_size:]
        
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # 编码序列
        encoded = self.tokenizer(
            sequence,
            max_length=MAX_SEQUENCE_LENGTH,
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
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_TEST_SPLIT,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    train_dataset = ProteinDataset(split="train", data_path=DATA_PATH)
    val_dataset = ProteinDataset(split="val", data_path=DATA_PATH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ProteinDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ProteinDataset.collate_fn
    )
    
    return train_loader, val_loader 