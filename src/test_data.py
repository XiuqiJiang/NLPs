import os
import sys
import torch
from torch.utils.data import DataLoader
from src.utils.data_utils import create_data_loaders, load_sequences
from config.config import (
    SEQUENCE_FILE,
    EMBEDDING_FILE,
    BATCH_SIZE,
    TRAIN_TEST_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
    MAX_SEQUENCE_LENGTH
)

def test_data_loading():
    """测试数据加载过程"""
    print("="*50)
    print("开始测试数据加载...")
    print("="*50)
    
    # 1. 检查原始序列文件
    print("\n1. 检查原始序列文件:")
    print(f"序列文件路径: {SEQUENCE_FILE} (已切换为3~5C)")
    if os.path.exists(SEQUENCE_FILE):
        print("文件存在")
        sequences = load_sequences(SEQUENCE_FILE)
        print(f"加载的序列数量: {len(sequences)}")
        print("\n前5个序列示例:")
        for i, seq in enumerate(sequences[:5]):
            print(f"序列 {i+1}: {seq}")
    else:
        print("错误: 序列文件不存在!")
        return
    
    # 2. 检查预处理数据文件
    print("\n2. 检查预处理数据文件:")
    print(f"预处理数据文件路径: {EMBEDDING_FILE}")
    if os.path.exists(EMBEDDING_FILE):
        print("预处理数据文件存在")
        data = torch.load(EMBEDDING_FILE)
        print(f"数据文件中的键: {list(data.keys())}")
        print(f"Embeddings 形状: {data['embeddings'].shape}")
        print(f"序列数量: {len(data['sequences'])}")
    else:
        print("预处理数据文件不存在，将创建新的数据加载器")
    
    # 3. 创建数据加载器
    print("\n3. 创建数据加载器:")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"训练集/验证集比例: {TRAIN_TEST_SPLIT}")
    
    train_loader, val_loader = create_data_loaders(
        data_file=EMBEDDING_FILE,
        batch_size=BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    )
    
    # 4. 检查数据集和加载器的大小
    print("\n4. 数据集和加载器信息:")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 5. 检查一个批次的数据
    print("\n5. 检查一个批次的数据:")
    for batch in train_loader:
        print("\n批次数据形状:")
        for key, value in batch.items():
            print(f"{key}: {value.shape}")
        break  # 只检查第一个批次
    
    print("\n" + "="*50)
    print("数据加载测试完成")
    print("="*50)

if __name__ == "__main__":
    test_data_loading() 