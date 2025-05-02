# src/utils/data_utils.py (或者你的文件路径)

import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
# 确保导入了正确的配置文件路径
# from config.data_config import MAX_SEQUENCE_LENGTH, ALPHABET, NUM_WORKERS, PIN_MEMORY
# from config.config import ESM_MODEL_NAME, EMBEDDING_FILE, SEQUENCE_FILE # 假设这些在 config.py 中
import pandas as pd
from tqdm import tqdm # 建议添加 tqdm 以显示进度

# --- 从你的 config 文件导入 ---
# 假设你的 config.py 和 config/data_config.py 定义了这些变量
# 如果没有，你需要在这里定义它们或从正确的地方导入
try:
    from config.data_config import (
        MAX_SEQUENCE_LENGTH,
        ALPHABET,
        SPECIAL_TOKENS,
        VOCAB_SIZE,
        PAD_TOKEN_ID,
        EOS_TOKEN_ID,
        NUM_WORKERS,
        PIN_MEMORY
    )
except ImportError:
    print("Warning: Could not import from config.data_config. Using default values.")
    MAX_SEQUENCE_LENGTH = 1024
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    SPECIAL_TOKENS = ['<pad>', '<eos>']
    VOCAB_SIZE = len(ALPHABET) + len(SPECIAL_TOKENS)
    PAD_TOKEN_ID = len(ALPHABET)
    EOS_TOKEN_ID = len(ALPHABET) + 1
    NUM_WORKERS = 0
    PIN_MEMORY = False

# 创建字符到ID的映射
CHAR_TO_ID = {char: idx for idx, char in enumerate(ALPHABET)}
for idx, token in enumerate(SPECIAL_TOKENS):
    CHAR_TO_ID[token] = len(ALPHABET) + idx

# 创建ID到字符的映射
ID_TO_CHAR = {idx: char for char, idx in CHAR_TO_ID.items()}

def sequence_to_ids(sequence: str) -> List[int]:
    """将氨基酸序列转换为ID序列，并添加EOS token"""
    ids = [CHAR_TO_ID[char] for char in sequence]
    ids.append(EOS_TOKEN_ID)  # 添加EOS token
    return ids

def ids_to_sequence(ids: List[int]) -> str:
    """将ID序列转换回氨基酸序列，忽略特殊token"""
    sequence = []
    for id in ids:
        if id < len(ALPHABET):  # 只处理氨基酸token
            sequence.append(ID_TO_CHAR[id])
        elif id == EOS_TOKEN_ID:  # 遇到EOS token就停止
            break
    return ''.join(sequence)

def pad_sequence(ids: List[int], max_length: int = MAX_SEQUENCE_LENGTH) -> List[int]:
    """对ID序列进行填充"""
    if len(ids) > max_length:
        ids = ids[:max_length-1] + [EOS_TOKEN_ID]  # 截断并确保以EOS结束
    else:
        ids = ids + [PAD_TOKEN_ID] * (max_length - len(ids))  # 填充到最大长度
    return ids

try:
    from config.config import (
        ESM_MODEL_NAME, # 例如 "facebook/esm2_t6_8M_UR50D"
        EMBEDDING_FILE, # 例如 "data/processed/embeddings.pt"
        SEQUENCE_FILE, # 例如 "data/raw/sequences.txt"
        VAE_BATCH_SIZE, # 例如 64
        ESM_MODEL_PATH, # 例如 "path/to/your/esm_model"
        DEVICE # 例如 "cuda" 或 "cpu"
    )
except ImportError:
     print("Warning: Could not import from config.config. Using default values.")
     ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # 提供一个默认值
     EMBEDDING_FILE = "embeddings.pt"
     SEQUENCE_FILE = "sequences.txt"
     VAE_BATCH_SIZE = 64
     ESM_MODEL_PATH = "path/to/your/esm_model"
     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Dataset 类 ---
class EmbeddingDataset(Dataset):
    """使用预计算 Embeddings 的数据集"""

    def __init__(
        self,
        embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        sequences: List[str],
        max_length: int = MAX_SEQUENCE_LENGTH
    ) -> None:
        """初始化数据集

        Args:
            embeddings: 预计算的 ESM embeddings (应为 [N, seq_len, embed_dim])
            attention_masks: 对应的 attention masks (应为 [N, seq_len])
            sequences: 对应的原始序列字符串列表
            max_length: 序列的最大长度
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"embeddings 必须是 torch.Tensor, 但收到了 {type(embeddings)}")
        if not isinstance(attention_masks, torch.Tensor):
            raise TypeError(f"attention_masks 必须是 torch.Tensor, 但收到了 {type(attention_masks)}")
        if len(embeddings) != len(sequences) or len(attention_masks) != len(sequences):
            raise ValueError(f"Embeddings ({len(embeddings)}), attention masks ({len(attention_masks)}) 和 sequences ({len(sequences)}) 的数量必须匹配")

        # 确保 embeddings 和 attention masks 在 CPU 上
        self.embeddings = embeddings.cpu().float()
        self.attention_masks = attention_masks.cpu()
        self.sequences = sequences
        self.max_length = max_length

        # 验证 embeddings 和 attention masks 的形状
        if self.embeddings.ndim != 3:
            raise ValueError(f"Embeddings 的维度必须是 3 [N, seq_len, embed_dim], 但收到了 {self.embeddings.shape}")
        if self.attention_masks.ndim != 2:
            raise ValueError(f"Attention masks 的维度必须是 2 [N, seq_len], 但收到了 {self.attention_masks.shape}")
        if self.embeddings.shape[:2] != self.attention_masks.shape:
            raise ValueError(f"Embeddings 和 attention masks 的前两个维度必须匹配: {self.embeddings.shape[:2]} != {self.attention_masks.shape}")

        # 为每个序列生成 target IDs
        self.target_ids = []
        for seq in sequences:
            # 将序列转换为 IDs 并添加 EOS token
            ids = sequence_to_ids(seq)
            # 填充到最大长度
            padded_ids = pad_sequence(ids, max_length)
            self.target_ids.append(padded_ids)

        self.target_ids = torch.tensor(self.target_ids, dtype=torch.long)

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
            - 'target_ids': 形状为 [seq_len] 的 tensor
        """
        return {
            'embeddings': self.embeddings[idx],  # [seq_len, embed_dim]
            'attention_mask': self.attention_masks[idx],  # [seq_len]
            'target_ids': self.target_ids[idx]  # [seq_len]
        }

# --- 数据加载函数 ---
def load_sequences(file_path: str) -> List[str]:
    """从文本文件加载蛋白质序列 (每行一个，忽略 '>' 开头的行)"""


    sequences = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('>'):
                    # 基本的氨基酸字符检查 (可选, 但推荐)
                    # valid_chars = set("ACDEFGHIKLMNPQRSTVWYUOXBJZ") # UOXBJZ 是非标准/模糊的
                    # if all(c.upper() in valid_chars for c in line):
                    sequences.append(line.upper()) # 转换为大写
                    # else:
                    #    print(f"警告: 跳过包含无效字符的序列: {line[:50]}...")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        raise

    if not sequences:
        print(f"警告: 文件 {file_path} 中没有找到任何有效序列")

    print(f"从文件 {file_path} 中成功加载 {len(sequences)} 个序列")
    return sequences

# --- Embedding 生成函数 (修正了池化逻辑) ---
def get_esm_embeddings(
    sequences: List[str],
    model_path: str = ESM_MODEL_PATH,
    device: str = DEVICE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """使用ESM模型获取序列的嵌入表示
    
    Args:
        sequences: 氨基酸序列列表
        model_path: ESM模型路径
        device: 设备
        
    Returns:
        (embeddings, attention_mask)
        embeddings: 形状为 [batch_size, seq_len, embedding_dim]
        attention_mask: 形状为 [batch_size, seq_len]
    """
    # 加载模型和tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 对序列进行编码
    inputs = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    # 获取注意力掩码
    attention_mask = inputs['attention_mask']
    
    # 获取嵌入表示
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层的隐藏状态作为嵌入
        embeddings = outputs.hidden_states[-1]
    
    return embeddings, attention_mask

# --- DataLoader 创建函数 ---
def create_data_loaders(
    data_path: str,
    batch_size: int,
    train_test_split: float = 0.15, # 注意：这里是验证集比例
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器

    Args:
        data_path: 预计算数据的文件路径
        batch_size: 批处理大小
        train_test_split: 验证集比例
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中
        max_sequence_length: 序列的最大长度

    Returns:
        训练和验证数据加载器
    """
    # 加载数据以获取总样本数
    data = torch.load(data_path)
    num_samples = len(data['embeddings'])
    
    print(f"创建 DataLoaders: 总样本数={num_samples}")
    
    # 划分训练集和验证集索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices) # 原地打乱
    split_idx = int(num_samples * (1 - train_test_split)) # 训练集结束的索引
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"训练集样本数: {len(train_indices)}, 验证集样本数: {len(val_indices)}")
    
    # 创建数据集实例
    print("创建训练数据集...")
    train_dataset = ProteinDataset(
        data_path=data_path,
        max_length=max_sequence_length
    )
    # 只使用训练集索引
    train_dataset.embeddings = train_dataset.embeddings[train_indices]
    train_dataset.attention_masks = train_dataset.attention_masks[train_indices]
    train_dataset.token_ids = train_dataset.token_ids[train_indices]
    train_dataset.sequences = [train_dataset.sequences[i] for i in train_indices]
    
    print("创建验证数据集...")
    val_dataset = ProteinDataset(
        data_path=data_path,
        max_length=max_sequence_length
    )
    # 只使用验证集索引
    val_dataset.embeddings = val_dataset.embeddings[val_indices]
    val_dataset.attention_masks = val_dataset.attention_masks[val_indices]
    val_dataset.token_ids = val_dataset.token_ids[val_indices]
    val_dataset.sequences = [val_dataset.sequences[i] for i in val_indices]
    
    # 创建数据加载器
    print("创建 DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # 训练时打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,        # 丢弃最后一个不完整的批次
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # 验证时不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,       # 验证时通常不丢弃
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )
    
    print("DataLoaders 创建完成。")
    return train_loader, val_loader

# --- VAE 数据准备主函数 ---
def prepare_VAE_data(
    data_source: str | List[str], # 文件路径或序列列表
    embedding_path: str = EMBEDDING_FILE, # 保存/加载 embedding 的路径
    force_regenerate: bool = False, # 是否强制重新生成 embedding
    is_csv: bool = False, # 如果 data_source 是路径，指示是否为 csv
    csv_column: str = 'sequence' # CSV 文件中包含序列的列名
) -> Tuple[torch.Tensor, List[str]]:
    """准备 VAE 的输入数据 (加载/生成 Embeddings 和序列)"""

    sequences = []
    # 1. 加载序列
    if isinstance(data_source, str):
        print(f"从路径加载序列: {data_source}")

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
    valid_chars = set(ALPHABET[2:])  # 排除特殊token
    for i, seq in enumerate(sequences):
        invalid_chars = set(seq) - valid_chars
        if invalid_chars:
            raise ValueError(f"序列 {i} 包含无效字符: {invalid_chars}")
    
    # 加载模型和tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_path).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 对序列进行编码
    inputs = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(DEVICE)
    
    # 获取注意力掩码
    attention_masks = inputs['attention_mask']
    
    # 获取嵌入表示
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层的隐藏状态作为嵌入
        embeddings = outputs.hidden_states[-1]
    
    # 生成 token IDs
    token_ids = []
    for seq in sequences:
        # 将序列转换为 IDs 并添加 EOS token
        ids = sequence_to_ids(seq)
        # 填充到最大长度
        padded_ids = pad_sequence(ids, max_length)
        token_ids.append(padded_ids)
    
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    
    # 验证数据形状
    print(f"验证数据形状:")
    print(f"Embeddings: {embeddings.shape}")  # [batch_size, seq_len, embed_dim]
    print(f"Attention masks: {attention_masks.shape}")  # [batch_size, seq_len]
    print(f"Token IDs: {token_ids.shape}")  # [batch_size, seq_len]
    
    # 验证 token IDs 的有效性
    assert token_ids.max() < VOCAB_SIZE, f"Token IDs 包含超出词汇表大小的值: {token_ids.max()} >= {VOCAB_SIZE}"
    assert token_ids.min() >= 0, f"Token IDs 包含负值: {token_ids.min()}"
    
    # 验证序列长度
    assert embeddings.shape[1] == max_length, f"Embeddings 序列长度 ({embeddings.shape[1]}) 与 max_length ({max_length}) 不匹配"
    assert attention_masks.shape[1] == max_length, f"Attention masks 序列长度 ({attention_masks.shape[1]}) 与 max_length ({max_length}) 不匹配"
    assert token_ids.shape[1] == max_length, f"Token IDs 序列长度 ({token_ids.shape[1]}) 与 max_length ({max_length}) 不匹配"
    
    # 验证特殊token
    for i, ids in enumerate(token_ids):
        if ids[-1] != PAD_TOKEN_ID:
            print(f"警告: 序列 {i} 的最后一个token不是PAD token")
        if EOS_TOKEN_ID not in ids:
            print(f"警告: 序列 {i} 中没有EOS token")
    
    # 保存到文件
    data = {
        'embeddings': embeddings.cpu(),
        'attention_masks': attention_masks.cpu(),
        'token_ids': token_ids.cpu(),
        'sequences': sequences,
        'config': {
            'max_length': max_length,
            'vocab_size': VOCAB_SIZE,
            'pad_token_id': PAD_TOKEN_ID,
            'eos_token_id': EOS_TOKEN_ID
        }
    }
    
    torch.save(data, output_path)
    print(f"数据已保存到 {output_path}")
    print(f"配置信息:")
    print(f"- 最大序列长度: {max_length}")
    print(f"- 词汇表大小: {VOCAB_SIZE}")
    print(f"- Padding token ID: {PAD_TOKEN_ID}")
    print(f"- EOS token ID: {EOS_TOKEN_ID}")

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
        if config['vocab_size'] != VOCAB_SIZE:
            raise ValueError(f"数据文件中的词汇表大小 ({config['vocab_size']}) 与配置的词汇表大小 ({VOCAB_SIZE}) 不匹配")
        if config['pad_token_id'] != PAD_TOKEN_ID:
            raise ValueError(f"数据文件中的PAD token ID ({config['pad_token_id']}) 与配置的PAD token ID ({PAD_TOKEN_ID}) 不匹配")
        if config['eos_token_id'] != EOS_TOKEN_ID:
            raise ValueError(f"数据文件中的EOS token ID ({config['eos_token_id']}) 与配置的EOS token ID ({EOS_TOKEN_ID}) 不匹配")
        
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
