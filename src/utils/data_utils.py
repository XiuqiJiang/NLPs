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
    """蛋白质序列数据集，从预计算的文件加载数据"""

    def __init__(
        self,
        data_path: str,
        max_length: int = MAX_SEQUENCE_LENGTH
    ) -> None:
        """初始化数据集
        
        Args:
            data_path: 预计算数据的文件路径
            max_length: 序列的最大长度
        """
        # 验证最大长度
        if max_length <= 0:
            raise ValueError(f"最大长度必须大于0，但收到了 {max_length}")
        if max_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"最大长度不能超过配置的最大长度 {MAX_SEQUENCE_LENGTH}，但收到了 {max_length}")
        
        # 加载数据
        data = torch.load(data_path)
        
        # 验证数据完整性
        required_keys = ['embeddings', 'attention_masks', 'token_ids', 'sequences', 'config']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"数据文件缺少必需的键: {key}")
        
        # 验证配置一致性
        config = data['config']
        if config['max_length'] != max_length:
            raise ValueError(f"数据文件中的最大长度 ({config['max_length']}) 与指定的最大长度 ({max_length}) 不匹配")
        
        # 确保数据在 CPU 上
        self.embeddings = data['embeddings'].cpu().float()
        self.attention_masks = data['attention_masks'].cpu()
        self.token_ids = data['token_ids'].cpu()
        self.sequences = data['sequences']
        self.max_length = max_length
        
        # 验证数据形状
        if self.embeddings.ndim != 3:
            raise ValueError(f"Embeddings 的维度必须是 3 [N, seq_len, embed_dim], 但收到了 {self.embeddings.shape}")
        if self.attention_masks.ndim != 2:
            raise ValueError(f"Attention masks 的维度必须是 2 [N, seq_len], 但收到了 {self.attention_masks.shape}")
        if self.token_ids.ndim != 2:
            raise ValueError(f"Token IDs 的维度必须是 2 [N, seq_len], 但收到了 {self.token_ids.shape}")
        if len(self.embeddings) != len(self.sequences):
            raise ValueError(f"Embeddings ({len(self.embeddings)}) 和 sequences ({len(self.sequences)}) 数量不匹配")
        
        # 验证序列长度
        if self.embeddings.shape[1] != max_length:
            raise ValueError(f"Embeddings 序列长度 ({self.embeddings.shape[1]}) 与最大长度 ({max_length}) 不匹配")
        if self.attention_masks.shape[1] != max_length:
            raise ValueError(f"Attention masks 序列长度 ({self.attention_masks.shape[1]}) 与最大长度 ({max_length}) 不匹配")
        if self.token_ids.shape[1] != max_length:
            raise ValueError(f"Token IDs 序列长度 ({self.token_ids.shape[1]}) 与最大长度 ({max_length}) 不匹配")
        
        print(f"ProteinDataset 初始化: {len(self.embeddings)} 个样本")
        print(f"Embeddings 形状: {self.embeddings.shape}")
        print(f"Attention masks 形状: {self.attention_masks.shape}")
        print(f"Token IDs 形状: {self.token_ids.shape}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含以下键的字典:
            - 'embeddings': 形状为 [seq_len, embed_dim] 的 tensor
            - 'attention_mask': 形状为 [seq_len] 的 tensor
            - 'token_ids': 形状为 [seq_len] 的 tensor
            - 'sequence_id': 序列的索引
        """
        return {
            'embeddings': self.embeddings[idx],  # [seq_len, embed_dim]
            'attention_mask': self.attention_masks[idx],  # [seq_len]
            'token_ids': self.token_ids[idx],  # [seq_len]
            'sequence_id': idx
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
    batch_size: int = 32,
    train_test_split: float = 0.1,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器
    
    Args:
        data_file: 预计算数据的文件路径
        batch_size: 批处理大小
        train_test_split: 验证集比例
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将张量固定在内存中
        max_sequence_length: 序列的最大长度
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    # 直接使用 data_file 路径创建数据集
    dataset = ProteinDataset(
        data_path=data_file,
        max_length=max_sequence_length
    )
    
    # 计算训练集和验证集的大小
    dataset_size = len(dataset)
    # 确保至少有一个验证样本
    val_size = max(1, int(train_test_split * dataset_size))
    train_size = dataset_size - val_size
    
    if train_size <= 0:
        raise ValueError("训练集大小计算后小于等于0，请检查 train_test_split 或数据集大小")
    
    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
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
    
    print(f"创建 DataLoaders: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    return train_loader, val_loader

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """将一批数据项组合成一个批次
    
    Args:
        batch: 数据项列表
        
    Returns:
        包含以下键的字典:
        - 'embeddings': 形状为 [batch_size, seq_len, embed_dim] 的 tensor
        - 'attention_mask': 形状为 [batch_size, seq_len] 的 tensor
        - 'token_ids': 形状为 [batch_size, seq_len] 的 tensor
        - 'sequence_ids': 形状为 [batch_size] 的 tensor
    """
    # 获取批次大小
    batch_size = len(batch)
    
    # 获取序列长度和嵌入维度
    seq_len = batch[0]['embeddings'].shape[0]
    embed_dim = batch[0]['embeddings'].shape[1]
    
    # 初始化批次张量
    embeddings = torch.zeros((batch_size, seq_len, embed_dim))
    attention_masks = torch.zeros((batch_size, seq_len), dtype=torch.long)
    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    sequence_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # 填充批次张量
    for i, item in enumerate(batch):
        embeddings[i] = item['embeddings']
        attention_masks[i] = item['attention_mask']
        token_ids[i] = item['token_ids']
        sequence_ids[i] = item['sequence_id']
    
    return {
        'embeddings': embeddings,
        'attention_mask': attention_masks,
        'token_ids': token_ids,
        'sequence_ids': sequence_ids
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
    """预处理序列并生成 embeddings、masks 和 token IDs，保存到文件
    
    Args:
        sequences: 氨基酸序列列表
        output_path: 输出文件路径
        model_path: ESM 模型路径
        batch_size: 批处理大小
        max_length: 最大序列长度（包括特殊token）
    """
    print(f"开始预处理 {len(sequences)} 个序列...")
    
    # 验证序列长度
    for i, seq in enumerate(sequences):
        if len(seq) > max_length - 1:  # -1 是为了留出EOS token的位置
            print(f"警告: 序列 {i} 长度 ({len(seq)}) 超过最大长度 ({max_length-1})，将被截断")
    
    # 验证序列字符
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYUOXBJZ")  # 排除特殊token
    for i, seq in enumerate(sequences):
        invalid_chars = set(seq) - valid_chars
        if invalid_chars:
            raise ValueError(f"序列 {i} 包含无效字符: {invalid_chars}")
    
    # 加载模型和tokenizer
    print(f"从 {model_path} 加载模型...")
    try:
        # 首先尝试从本地加载
        model = AutoModelForMaskedLM.from_pretrained(
            model_path,
            local_files_only=True
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # 验证模型和tokenizer的词汇表大小是否匹配
        model_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = tokenizer.vocab_size
        print(f"模型词汇表大小: {model_vocab_size}")
        print(f"Tokenizer词汇表大小: {tokenizer_vocab_size}")
        
        if model_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                f"模型词汇表大小 ({model_vocab_size}) 与tokenizer词汇表大小 ({tokenizer_vocab_size}) 不匹配"
            )
            
    except Exception as e:
        print(f"从本地加载失败: {str(e)}")
        raise
    
    # 对序列进行编码
    print("对序列进行编码...")
    # 确保使用正确的填充和截断设置
    inputs = tokenizer(
        sequences,
        padding='max_length',  # 使用max_length填充
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,  # 确保返回attention mask
        return_token_type_ids=False  # ESM模型不需要token_type_ids
    ).to(DEVICE)
    
    # 验证输入形状
    print(f"输入形状:")
    print(f"input_ids: {inputs['input_ids'].shape}")
    print(f"attention_mask: {inputs['attention_mask'].shape}")
    
    # 验证token IDs的有效性
    if inputs['input_ids'].max() >= tokenizer.vocab_size:
        raise ValueError(
            f"Token IDs包含超出词汇表大小的值: "
            f"{inputs['input_ids'].max()} >= {tokenizer.vocab_size}"
        )
    if inputs['input_ids'].min() < 0:
        raise ValueError(f"Token IDs包含负值: {inputs['input_ids'].min()}")
    
    # 获取注意力掩码和token_ids
    attention_masks = inputs['attention_mask']
    token_ids = inputs['input_ids']  # 直接使用tokenizer生成的token_ids
    
    # 获取嵌入表示
    print("生成嵌入表示...")
    with torch.no_grad():
        # 确保所有输入都在正确的设备上
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        outputs = model(**model_inputs, output_hidden_states=True)
        
        # 使用最后一层的隐藏状态作为嵌入
        embeddings = outputs.hidden_states[-1]
        
        # 验证输出形状
        print(f"模型输出形状:")
        print(f"last_hidden_state: {embeddings.shape}")
    
    # 确保embeddings的长度符合要求
    if embeddings.shape[1] != max_length:
        print(f"警告: embeddings长度 ({embeddings.shape[1]}) 与max_length ({max_length}) 不匹配")
        print("检查模型配置和输入...")
        
        # 检查模型配置
        print(f"模型配置:")
        print(f"- 最大位置嵌入: {model.config.max_position_embeddings}")
        print(f"- 隐藏层大小: {model.config.hidden_size}")
        print(f"- 模型类型: {model.__class__.__name__}")
        print(f"- 词汇表大小: {model.config.vocab_size}")
        
        # 检查输入
        print(f"输入检查:")
        print(f"- input_ids 形状: {inputs['input_ids'].shape}")
        print(f"- attention_mask 形状: {inputs['attention_mask'].shape}")
        print(f"- attention_mask 非零元素数: {(inputs['attention_mask'] > 0).sum(dim=1)}")
        print(f"- input_ids 中的特殊token数量: {(inputs['input_ids'] >= tokenizer.vocab_size).sum(dim=1)}")
        
        # 创建新的填充后的embeddings
        print("创建填充后的embeddings...")
        padded_embeddings = torch.zeros(
            (embeddings.shape[0], max_length, embeddings.shape[2]),
            dtype=embeddings.dtype,
            device=embeddings.device
        )
        # 复制原始embeddings
        padded_embeddings[:, :embeddings.shape[1], :] = embeddings
        embeddings = padded_embeddings
    
    # 验证数据形状
    print(f"验证数据形状:")
    print(f"Embeddings: {embeddings.shape}")  # [batch_size, seq_len, embed_dim]
    print(f"Attention masks: {attention_masks.shape}")  # [batch_size, seq_len]
    print(f"Token IDs: {token_ids.shape}")  # [batch_size, seq_len]
    
    # 验证 token IDs 的有效性
    assert token_ids.max() < tokenizer.vocab_size, f"Token IDs 包含超出词汇表大小的值: {token_ids.max()} >= {tokenizer.vocab_size}"
    assert token_ids.min() >= 0, f"Token IDs 包含负值: {token_ids.min()}"
    
    # 验证序列长度
    assert embeddings.shape[1] == max_length, f"Embeddings 序列长度 ({embeddings.shape[1]}) 与 max_length ({max_length}) 不匹配"
    assert attention_masks.shape[1] == max_length, f"Attention masks 序列长度 ({attention_masks.shape[1]}) 与 max_length ({max_length}) 不匹配"
    assert token_ids.shape[1] == max_length, f"Token IDs 序列长度 ({token_ids.shape[1]}) 与 max_length ({max_length}) 不匹配"
    
    # 验证特殊token
    for i, ids in enumerate(token_ids):
        if ids[-1] != tokenizer.pad_token_id:
            print(f"警告: 序列 {i} 的最后一个token不是PAD token")
        if tokenizer.eos_token_id not in ids:
            print(f"警告: 序列 {i} 中没有EOS token")
    
    # 保存到文件
    print(f"保存数据到 {output_path}...")
    data = {
        'embeddings': embeddings.cpu(),
        'attention_masks': attention_masks.cpu(),
        'token_ids': token_ids.cpu(),
        'sequences': sequences,
        'config': {
            'max_length': max_length,
            'vocab_size': tokenizer.vocab_size,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'model_name': model.__class__.__name__,
            'embedding_dim': embeddings.shape[2],
            'model_path': model_path  # 保存模型路径以便验证
        }
    }
    
    torch.save(data, output_path)
    print(f"数据已保存到 {output_path}")
    print(f"配置信息:")
    print(f"- 最大序列长度: {max_length}")
    print(f"- 词汇表大小: {tokenizer.vocab_size}")
    print(f"- Padding token ID: {tokenizer.pad_token_id}")
    print(f"- EOS token ID: {tokenizer.eos_token_id}")
    print(f"- 模型类型: {model.__class__.__name__}")
    print(f"- 嵌入维度: {embeddings.shape[2]}")
    print(f"- 模型路径: {model_path}")
