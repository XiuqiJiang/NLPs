import torch
import numpy as np
from transformers import AutoTokenizer
from config.config import *
from models.vae import ESMVAE
from data.dataset import load_sequences, create_data_loaders
from utils.trainer import train_model

def main():
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 加载数据
    sequences = load_sequences(DATA_PATH)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(sequences, tokenizer)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESMVAE().to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)
    
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=VAE_EPOCHS,
        model_save_path=MODEL_WEIGHTS_PATH,
        kl_weight=KL_WEIGHT
    )

if __name__ == '__main__':
    main() 