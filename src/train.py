import os
import torch
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime

from config.model_config import *
from config.train_config import *
from config.data_config import *
from src.models.vae import ESMVAE
from src.utils.data_utils import load_sequences, create_data_loaders
from src.utils.trainer import VAETrainer

def main() -> None:
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("开始训练...")
    
    # 加载数据
    print("加载数据...")
    sequences = load_sequences(RAW_DATA_PATH)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        sequences,
        tokenizer,
        batch_size=VAE_BATCH_SIZE,
        train_test_split=TRAIN_TEST_SPLIT,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
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