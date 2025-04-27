import torch
import numpy as np
from tqdm import tqdm
import os
from config.config import *

def train_epoch(model, train_loader, optimizer, device, kl_weight=KL_WEIGHT):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        recon_x, mean, logvar = model(input_ids)
        
        # 计算损失
        loss, recon_loss, kl_loss = vae_loss(recon_x, input_ids, mean, logvar, kl_weight)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
    
    return train_loss / len(train_loader), train_recon_loss / len(train_loader), train_kl_loss / len(train_loader)

def validate(model, val_loader, device, kl_weight=KL_WEIGHT):
    """验证模型"""
    model.eval()
    val_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            recon_x, mean, logvar = model(input_ids)
            loss, recon_loss, kl_loss = vae_loss(recon_x, input_ids, mean, logvar, kl_weight)
            
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
    
    return val_loss / len(val_loader), val_recon_loss / len(val_loader), val_kl_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=VAE_EPOCHS, 
                model_save_path=MODEL_WEIGHTS_PATH, kl_weight=KL_WEIGHT):
    """训练模型的主函数"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, device, kl_weight
        )
        
        # 验证
        val_loss, val_recon_loss, val_kl_loss = validate(
            model, val_loader, device, kl_weight
        )
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})')
        print(f'Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f})')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved at {model_save_path}') 