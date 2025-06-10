# src/utils/data_utils.py (或者你的文件路径)

import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from .ring_utils import get_ring_info

from config.config import (
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PIN_MEMORY,
    ESM_MODEL_NAME,
    EMBEDDING_FILE,
    SEQUENCE_FILE,
    ESM_MODEL_PATH,
    DEVICE,
    RANDOM_SEED
)

# --- Dataset 类 ---
class ProteinDataset(Dataset):
    """蛋白质序列数据集"""
    
    def __init__(
        self,
        sequences: List[str],
        embeddings: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_length: int
    ):
        """初始化数据集
        
        Args:
            sequences: 序列列表
            embeddings: ESM嵌入张量
            tokenizer: tokenizer
            max_length: 最大序列长度
        """
        self.sequences = sequences
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 计算每个序列的环数
        self.ring_info = torch.tensor([get_ring_info(seq) for seq in sequences], dtype=torch.long)
        
        # 记录环数分布
        ring_dist = {}
        for rings in self.ring_info:
            ring_dist[rings.item()] = ring_dist.get(rings.item(), 0) + 1
        logging.info(f"环数分布: {ring_dist}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            包含以下键的字典:
            - input_ids: token IDs [seq_len]
            - attention_mask: 注意力掩码 [seq_len]
            - embeddings: ESM嵌入 [seq_len, embed_dim]
            - ring_info: 环数信息 [1]
        """
        sequence = self.sequences[idx]
        embedding = self.embeddings[idx]  # [seq_len, embed_dim]
        ring_info = self.ring_info[idx]  # 标量值
        
        # 编码序列
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 确保所有张量的维度正确
        attention_mask = encoding['attention_mask'].squeeze(0)  # [seq_len]
        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        
        # 确保embedding的维度正确
        if embedding.dim() == 2:  # [seq_len, embed_dim]
            pass  # 保持原样
        elif embedding.dim() == 3:  # [1, seq_len, embed_dim]
            embedding = embedding.squeeze(0)  # 移除多余的维度
        else:
            raise ValueError(f"意外的embedding维度: {embedding.shape}")
        
        # 确保ring_info是标量
        if ring_info.dim() > 0:
            ring_info = ring_info.squeeze()
        
        return {
            'input_ids': input_ids,  # [seq_len]
            'attention_mask': attention_mask,  # [seq_len]
            'embeddings': embedding,  # [seq_len, embed_dim]
            'ring_info': ring_info  # 标量
        }

# --- 数据加载函数 ---
def load_sequences(file_path: str) -> List[str]:
    """从文本文件加载蛋白质序列 (每行一个，忽略 '>' 开头的行)"""
    sequences = []
    try:
        print(f"开始从文件 {file_path} 加载序列...")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"文件总行数: {len(lines)}")
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('>'):
                    sequences.append(line.upper())  # 转换为大写
                    if i < 5:  # 打印前5个序列作为示例
                        print(f"序列 {i+1}: {line}")
                
                if i % 50 == 0:  # 每50行打印一次进度
                    print(f"已处理 {i} 行...")
                    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        raise

    if not sequences:
        print(f"警告: 文件 {file_path} 中没有找到任何有效序列")
    else:
        print(f"从文件 {file_path} 中成功加载 {len(sequences)} 个序列")
        print(f"前5个序列示例:")
        for i, seq in enumerate(sequences[:5]):
            print(f"序列 {i+1}: {seq}")

    return sequences

# --- DataLoader 创建函数 ---
def create_data_loaders(
    data_file: str,
    batch_size: int,
    train_test_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_sequence_length: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器
    
    Args:
        data_file: 数据文件路径
        batch_size: 批次大小
        train_test_split: 训练集比例
        num_workers: 数据加载线程数
        pin_memory: 是否使用固定内存
        max_sequence_length: 最大序列长度
        
    Returns:
        (训练数据加载器, 验证数据加载器)
    """
    # 加载数据
    data = torch.load(data_file)
    # 兼容list[dict]格式
    if isinstance(data, list):
        # 兼容list[dict]格式
        sequences = [item['sequence'] for item in data]
        embeddings = torch.stack([item['embeddings'] for item in data])
    else:
        sequences = data['sequences']
        embeddings = data['embeddings']
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ESM_MODEL_PATH,  # 使用config.py中定义的路径，已指向Google Drive
        local_files_only=True
    )
    
    # 创建数据集
    dataset = ProteinDataset(
        sequences=sequences,
        embeddings=embeddings,
        tokenizer=tokenizer,
        max_length=max_sequence_length
    )
    
    # 划分训练集和验证集
    train_size = int((1 - train_test_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
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

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """将多个数据项组合成一个批次
    
    Args:
        batch: 数据项列表，每个数据项是一个字典
        
    Returns:
        包含以下键的字典:
        - embeddings: ESM嵌入 [batch_size, seq_len, embed_dim]
        - attention_mask: 注意力掩码 [batch_size, seq_len]
        - input_ids: token IDs [batch_size, seq_len]
        - ring_info: 环数信息 [batch_size]
    """
    # 获取序列长度和嵌入维度
    seq_len = batch[0]['embeddings'].shape[0]  # 从[seq_len, embed_dim]中获取
    embed_dim = batch[0]['embeddings'].shape[1]
    
    # 初始化批次张量
    batch_size = len(batch)
    embeddings = torch.zeros((batch_size, seq_len, embed_dim))
    attention_masks = torch.zeros((batch_size, seq_len), dtype=torch.long)
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    ring_info = torch.zeros(batch_size, dtype=torch.long)  # 修改为[batch_size]
    
    # 填充批次张量
    for i, item in enumerate(batch):
        embeddings[i] = item['embeddings']  # [seq_len, embed_dim]
        attention_masks[i] = item['attention_mask']  # [seq_len]
        input_ids[i] = item['input_ids']  # [seq_len]
        ring_info[i] = item['ring_info']  # 标量
    
    return {
        'embeddings': embeddings,  # [batch_size, seq_len, embed_dim]
        'attention_mask': attention_masks,  # [batch_size, seq_len]
        'input_ids': input_ids,  # [batch_size, seq_len]
        'ring_info': ring_info  # [batch_size]
    }

# --- VAE 数据准备主函数 ---
def prepare_VAE_data(
    data_source: str | List[str],
    embedding_path: str = EMBEDDING_FILE,
    force_regenerate: bool = False,
    is_csv: bool = False,
    csv_column: str = 'sequence'
) -> Tuple[torch.Tensor, List[str]]:
    """准备 VAE 的输入数据
    
    Args:
        data_source: 数据源（文件路径或序列列表）
        embedding_path: 保存/加载 embedding 的路径
        force_regenerate: 是否强制重新生成 embedding
        is_csv: 如果 data_source 是路径，指示是否为 csv
        csv_column: CSV 文件中包含序列的列名
        
    Returns:
        (embeddings, sequences): 嵌入向量和序列列表
    """
    sequences = []
    
    # 1. 加载序列
    if isinstance(data_source, str):
        print(f"从路径加载序列: {data_source}")
        if is_csv:
            # 从CSV文件加载
            df = pd.read_csv(data_source)
            if csv_column not in df.columns:
                raise ValueError(f"CSV文件中找不到列 '{csv_column}'")
            sequences = df[csv_column].tolist()
        else:
            # 从文本文件加载
            sequences = load_sequences(data_source)
    else:
        # 直接使用提供的序列列表
        sequences = data_source
    
    if not sequences:
        raise ValueError("没有找到有效的序列")
    
    # 2. 检查是否需要重新生成embeddings
    if not force_regenerate and os.path.exists(embedding_path):
        print(f"找到现有的embeddings文件: {embedding_path}")
        data = torch.load(embedding_path)
        return data['embeddings'], data['sequences']
    
    # 3. 生成新的embeddings
    print("生成新的embeddings...")
    preprocess_and_embed(
        sequences=sequences,
        output_path=embedding_path,
        max_length=MAX_SEQUENCE_LENGTH
    )
    
    # 4. 加载生成的embeddings
    data = torch.load(embedding_path)
    return data['embeddings'], data['sequences']

def preprocess_and_embed(
    sequences: List[str],
    output_path: str,
    model_path: str = ESM_MODEL_PATH,
    batch_size: int = 32,
    max_length: int = MAX_SEQUENCE_LENGTH
) -> None:
    """预处理序列并生成ESM嵌入
    
    Args:
        sequences: 序列列表
        output_path: 输出文件路径
        model_path: ESM模型路径
        batch_size: 批处理大小
        max_length: 最大序列长度
    """
    print(f"开始预处理 {len(sequences)} 个序列...")
    
    # 加载模型和tokenizer
    print(f"从 {model_path} 加载模型...")
    model = AutoModelForMaskedLM.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH, local_files_only=True)
    
    print(f"模型词汇表大小: {model.config.vocab_size}")
    print(f"Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 对序列进行编码
    print("对序列进行编码...")
    encodings = tokenizer(
        sequences,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    print("输入形状:")
    print(f"input_ids: {encodings['input_ids'].shape}")
    print(f"attention_mask: {encodings['attention_mask'].shape}")
    
    # 生成嵌入表示
    print("生成嵌入表示...")
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # 使用最后一层的隐藏状态
    
    print("模型输出形状:")
    print(f"last_hidden_state: {embeddings.shape}")
    
    # 检查维度
    if embeddings.shape[0] != len(sequences):
        print(f"警告: embeddings长度 ({embeddings.shape[0]}) 与序列数量 ({len(sequences)}) 不匹配")
    
    # 检查模型配置和输入
    print("检查模型配置和输入...")
    print("模型配置:")
    print(f"- 最大位置嵌入: {model.config.max_position_embeddings}")
    print(f"- 隐藏层大小: {model.config.hidden_size}")
    print(f"- 模型类型: {model.__class__.__name__}")
    print(f"- 词汇表大小: {model.config.vocab_size}")
    
    print("输入检查:")
    print(f"- input_ids 形状: {encodings['input_ids'].shape}")
    print(f"- attention_mask 形状: {encodings['attention_mask'].shape}")
    print(f"- attention_mask 非零元素数: {encodings['attention_mask'].sum(dim=1)}")
    print(f"- input_ids 中的特殊token数量: {(encodings['input_ids'] < 4).sum(dim=1)}")
    
    # 创建填充后的embeddings
    print("创建填充后的embeddings...")
    # 确保embeddings的维度正确
    if embeddings.shape[0] != len(sequences):
        # 如果embeddings的批次大小与序列数量不匹配，需要调整
        if embeddings.shape[0] > len(sequences):
            # 如果embeddings太大，截断它
            embeddings = embeddings[:len(sequences)]
        else:
            # 如果embeddings太小，用零填充
            padding = torch.zeros(
                (len(sequences) - embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]),
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            embeddings = torch.cat([embeddings, padding], dim=0)
    
    # 保存处理后的数据
    print("保存处理后的数据...")
    # 转换为 list[dict] 格式
    num_samples = embeddings.shape[0]
    samples = []
    for i in range(num_samples):
        sample = {
            'embeddings': embeddings[i],  # [seq_len, embed_dim]
            'attention_mask': encodings['attention_mask'][i],  # [seq_len]
            'input_ids': encodings['input_ids'][i],  # [seq_len]
            'sequence': sequences[i],
            'ring_info': get_ring_info(sequences[i])  # 新增ring_info字段
        }
        samples.append(sample)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存为标准格式
    torch.save(samples, output_path)
    print(f"已保存为标准 list[dict] 格式，样本数：{len(samples)}")
