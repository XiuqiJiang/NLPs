import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from config.config import *

class SequenceDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.embeddings[idx]
        }

def create_data_loaders(
    embeddings: torch.Tensor,
    batch_size: int = VAE_BATCH_SIZE,
    train_split: float = TRAIN_TEST_SPLIT,
    num_workers: int = 4,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    dataset = SequenceDataset(embeddings)
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