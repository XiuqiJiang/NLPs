# src/utils/data_utils.py (或者你的文件路径)

import os
import numpy as np
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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
        MAX_SEQUENCE_LENGTH, # 例如 1024
        ALPHABET, # 可能不需要了，因为使用 ESM tokenizer
        NUM_WORKERS, # 例如 4
        PIN_MEMORY # 例如 True
    )
except ImportError:
    print("Warning: Could not import from config.data_config. Using default values.")
    MAX_SEQUENCE_LENGTH = 1024
    NUM_WORKERS = 0
    PIN_MEMORY = False

try:
    from config.config import (
        ESM_MODEL_NAME, # 例如 "facebook/esm2_t6_8M_UR50D"
        EMBEDDING_FILE, # 例如 "data/processed/embeddings.pt"
        SEQUENCE_FILE, # 例如 "data/raw/sequences.txt"
        VAE_BATCH_SIZE # 例如 64
    )
except ImportError:
     print("Warning: Could not import from config.config. Using default values.")
     ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # 提供一个默认值
     EMBEDDING_FILE = "embeddings.pt"
     SEQUENCE_FILE = "sequences.txt"
     VAE_BATCH_SIZE = 64


# --- Dataset 类 ---
class EmbeddingDataset(Dataset):
    """使用预计算 Embeddings 的数据集"""

    def __init__(
        self,
        embeddings: torch.Tensor,
        sequences: List[str],
        max_length: int = 64 # 这个 max_length 用于 sequence_tensor, 与 ESM 的 max_length 可能不同
    ) -> None:
        """初始化数据集

        Args:
            embeddings: 预计算的 ESM embeddings (应为 [N, embed_dim])
            sequences: 对应的原始序列字符串列表
            max_length: 用于对 'sequence' 键进行编码/填充的最大长度 (如果需要的话)
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"embeddings 必须是 torch.Tensor, 但收到了 {type(embeddings)}")
        if len(embeddings) != len(sequences):
             raise ValueError(f"Embeddings ({len(embeddings)}) 和 sequences ({len(sequences)}) 的数量必须匹配")

        # 确保 embeddings 在 CPU 上，并且是 float 类型 (通常 embeddings 是 float)
        self.embeddings = embeddings.cpu().float()
        self.sequences = sequences
        self.max_length = max_length # 用于 'sequence' 键的填充长度

        # 验证 embeddings 的形状 (期望是二维 [N, embed_dim])
        if self.embeddings.ndim != 2:
             print(f"警告: Embeddings 的维度不是 2 (实际为 {self.embeddings.ndim}). "
                   f"期望形状为 [num_sequences, embedding_dim]. "
                   f"实际形状: {self.embeddings.shape}")
        else:
             print(f"EmbeddingDataset 初始化: {len(self.embeddings)} 个样本, "
                   f"Embedding 维度: {self.embeddings.shape[1]}")


    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据项

        Args:
            idx: 索引

        Returns:
            包含 'embeddings' (Tensor[embed_dim]) 和 'sequence' (Tensor[max_length]) 的字典。
            注意: 'sequence' 是使用 ord() 编码并填充的，可能不是 VAE 模型直接需要的。
        """
        # 获取预计算的 embedding (应为 1D Tensor [embed_dim])
        embedding = self.embeddings[idx]

        # --- 处理序列字符串 (可选, VAE可能只需要embedding) ---
        sequence_str = self.sequences[idx]
        # 使用 ord() 编码并填充/截断到 self.max_length
        # 注意：这种编码方式可能不是最优的
        sequence_tensor = torch.zeros(self.max_length, dtype=torch.long)
        try:
            # 确保 sequence_str 是字符串
            if not isinstance(sequence_str, str):
                 sequence_str = str(sequence_str) # 尝试转换
            sequence_encoded = torch.tensor([ord(c) for c in sequence_str], dtype=torch.long)
            seq_len = len(sequence_encoded)
            copy_len = min(seq_len, self.max_length)
            sequence_tensor[:copy_len] = sequence_encoded[:copy_len]
        except Exception as e:
            print(f"警告: 处理索引 {idx} 的序列 '{sequence_str[:50]}...' 时出错: {e}. 返回全零张量。")
            # 保留全零张量

        # --- 返回字典 ---
        # 确保 embedding 是 1D
        if embedding.ndim != 1:
             print(f"警告: 索引 {idx} 的 Embedding 维度不是 1 (实际为 {embedding.ndim}). "
                   f"形状: {embedding.shape}. 可能导致 collate 错误。")
             # 尝试修正: 如果是 [1, dim]，则 squeeze
             if embedding.shape[0] == 1 and embedding.ndim == 2:
                 embedding = embedding.squeeze(0)
             # 如果是其他奇怪形状，这里可能需要更复杂的处理或报错

        return {
            'embeddings': embedding,    # 形状: [embed_dim]
            'sequence': sequence_tensor # 形状: [max_length]
        }

# --- 数据加载函数 ---
def load_sequences(file_path: str) -> List[str]:
    """从文本文件加载蛋白质序列 (每行一个，忽略 '>' 开头的行)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"序列文件不存在: {file_path}")

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
    model_name: str = ESM_MODEL_NAME,
    batch_size: int = 32, # 根据 GPU 内存调整
    max_length: int = 64 # ESM 模型处理的最大长度
) -> torch.Tensor:
    """使用 ESM 模型获取序列的固定维度 embeddings (平均池化)"""
    print(f"开始使用模型 '{model_name}' 生成 embeddings...")
    # 加载 tokenizer 和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"加载模型或 Tokenizer '{model_name}' 时出错: {e}")
        raise

    # 将模型移到 GPU (如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型已移至 {device}")

    all_embeddings = []
    num_sequences = len(sequences)
    print(f"处理 {num_sequences} 个序列，批次大小: {batch_size}")

    # 使用 tqdm 显示进度条
    with tqdm(total=num_sequences, desc="生成 Embeddings") as pbar:
        # 分批处理序列
        for i in range(0, num_sequences, batch_size):
            batch_sequences = sequences[i:i + batch_size]
            current_batch_size = len(batch_sequences)

            # 对序列进行编码
            try:
                encoded = tokenizer(
                    batch_sequences,
                    padding=True,        # 填充到批次中最长序列
                    truncation=True,     # 截断超过 max_length 的序列
                    max_length=max_length,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"\n错误: Tokenizer 处理批次 {i//batch_size} 时出错: {e}")
                print(f"问题序列示例 (前50个字符): {[s[:50] for s in batch_sequences]}")
                # 可以选择跳过这个批次或停止
                # raise # 重新引发错误以停止
                print("警告: 跳过这个批次。")
                pbar.update(current_batch_size)
                continue # 跳到下一个批次

            # 将输入移到设备
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device) # 形状: [batch, seq_len]

            # 获取 embeddings
            with torch.no_grad():
                try:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # 获取最后一层的 hidden states, 形状: [batch, seq_len, embed_dim]
                    hidden_states = outputs.last_hidden_state

                    # --- **修正后的平均池化逻辑** ---
                    # 1. 获取 attention_mask (已经是 [batch, seq_len])
                    # 2. 将 mask 扩展以便与 hidden_states 相乘 (可选，也可以直接用 mask 索引)
                    #    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    # 3. 将 padding 位置的 hidden states 置零
                    #    hidden_states = hidden_states * mask_expanded
                    # 更简洁的方式：直接使用布尔索引置零 (可能效率稍低但更清晰)
                    hidden_states[attention_mask == 0] = 0

                    # 4. 沿着序列长度维度求和
                    #    形状: [batch, embed_dim]
                    summed_states = hidden_states.sum(dim=1)

                    # 5. 计算每个序列的实际长度 (非 padding token 的数量)
                    #    形状: [batch, 1] (使用 keepdim=True 以便广播)
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True)

                    # 6. 计算平均值 (防止除以零)
                    #    形状: [batch, embed_dim]
                    batch_embeddings = summed_states / seq_lengths.clamp(min=1)
                    # --- 池化结束 ---

                    # 将结果移回 CPU 并添加到列表
                    all_embeddings.append(batch_embeddings.cpu())

                except Exception as e:
                    print(f"\n错误: 模型前向传播或池化处理批次 {i//batch_size} 时出错: {e}")
                    # 可以添加更多调试信息，例如 input_ids 的形状
                    print(f"Input IDs shape: {input_ids.shape}")
                    # raise # 重新引发错误以停止
                    print("警告: 跳过这个批次。")

            pbar.update(current_batch_size) # 更新进度条

    if not all_embeddings:
        raise RuntimeError("未能生成任何 embeddings，请检查错误信息。")

    # 将所有批次的 embeddings 拼接起来
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings 生成完毕。最终形状: {embeddings_tensor.shape}") # 应该是 [num_sequences, embedding_dim]

    # 验证形状是否符合预期
    if embeddings_tensor.shape[0] != num_sequences:
        print(f"警告: 生成的 embeddings 数量 ({embeddings_tensor.shape[0]}) 与输入序列数量 ({num_sequences}) 不匹配！")
    if embeddings_tensor.ndim != 2:
         print(f"警告: 生成的 embeddings 张量维度不是 2 (实际为 {embeddings_tensor.ndim})！")

    return embeddings_tensor

# --- DataLoader 创建函数 ---
def create_data_loaders(
    embeddings: torch.Tensor,
    sequences: List[str],
    batch_size: int,
    train_test_split: float = 0.15, # 注意：这里是验证集比例
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    max_sequence_length_padding: int = 64 # 用于 EmbeddingDataset 中 'sequence' 键的填充
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"create_data_loaders 需要 torch.Tensor 类型的 embeddings, 收到 {type(embeddings)}")
    if embeddings.ndim != 2:
         raise ValueError(f"create_data_loaders 需要二维 embeddings [N, dim], 收到形状 {embeddings.shape}")
    if len(embeddings) != len(sequences):
         raise ValueError(f"Embeddings ({len(embeddings)}) 和 sequences ({len(sequences)}) 数量不匹配")

    print(f"创建 DataLoaders: 总样本数={len(embeddings)}, Embedding 维度={embeddings.shape[1]}")

    # 划分训练集和验证集索引
    indices = np.arange(len(embeddings))
    np.random.shuffle(indices) # 原地打乱
    split_idx = int(len(indices) * (1 - train_test_split)) # 训练集结束的索引

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    print(f"训练集样本数: {len(train_indices)}, 验证集样本数: {len(val_indices)}")

    # 创建数据集实例
    print("创建训练数据集...")
    train_dataset = EmbeddingDataset(
        embeddings=embeddings[train_indices], # 选择训练集的 embeddings
        sequences=[sequences[i] for i in train_indices], # 选择训练集的序列
        max_length=max_sequence_length_padding # 'sequence' 键的填充长度
    )
    print("创建验证数据集...")
    val_dataset = EmbeddingDataset(
        embeddings=embeddings[val_indices], # 选择验证集的 embeddings
        sequences=[sequences[i] for i in val_indices], # 选择验证集的序列
        max_length=max_sequence_length_padding # 'sequence' 键的填充长度
    )

    # 创建数据加载器
    print("创建 DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # 训练时打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True         # 丢弃最后一个不完整的批次，确保批次大小一致
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # 验证时不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False        # 验证时通常不丢弃，可以看到所有样本的结果 (除非需要固定大小)
                               # 如果验证逻辑也要求固定大小，可以设为 True
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
        if not os.path.exists(data_source):
             raise FileNotFoundError(f"数据源文件未找到
