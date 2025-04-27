import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config.config import *

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def load_sequences(file_path):
    """加载序列数据"""
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    return sequences

def create_data_loaders(sequences, tokenizer, batch_size=VAE_BATCH_SIZE, train_split=TRAIN_TEST_SPLIT):
    """创建训练和验证数据加载器"""
    dataset = SequenceDataset(sequences, tokenizer)
    train_size = int((1 - train_split) * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader 