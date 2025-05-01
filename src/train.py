import os
import torch
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime

from config.model_config import *
from config.train_config import *
from config.data_config import *
from src.models.vae import ESMVAE
from src.utils.data_utils import load_sequences, get_esm_embeddings, create_data_loaders
from src.utils.trainer import VAETrainer

def main() -> None:
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("开始训练...")
    
    # 准备数据
    print("准备数据...")
    file_path = "/content/NLPs/data/raw/NLP_sequences_no cesin.txt"
    print(f"正在加载序列文件: {file_path}")
    
    # 加载序列
    sequences = load_sequences(file_path)
    print(f"成功加载 {len(sequences)} 个序列")
    
    # 获取ESM embeddings
    print("正在计算ESM embeddings...")
    embeddings = get_esm_embeddings(sequences, batch_size=32)
    print(f"成功生成embeddings，形状: {embeddings.shape}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        embeddings=embeddings,
        sequences=sequences,
        batch_size=VAE_BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESMVAE().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)
    
    # 创建训练器
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        model_save_dir=MODEL_SAVE_DIR
    )
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=VAE_EPOCHS
    )

if __name__ == '__main__':
    main() 