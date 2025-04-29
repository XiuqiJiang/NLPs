import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from typing import Tuple, Optional
from config.model_config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIM,
    ALPHABET
)

class ESMVAE(nn.Module):
    """基于ESM的变分自编码器模型
    
    使用预训练的ESM模型作为编码器，实现蛋白质序列的变分自编码。
    
    Attributes:
        esm: 预训练的ESM模型
        encoder_mean: 编码器均值层
        encoder_logvar: 编码器方差层
        decoder: 解码器网络
        output_layer: 输出层
    """
    
    def __init__(
        self,
        esm_model: Optional[AutoModelForMaskedLM] = None,
        esm_model_name: str = ESM_MODEL_NAME,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM
    ) -> None:
        """初始化ESMVAE模型
        
        Args:
            esm_model: 预加载的ESM模型，如果为None则从esm_model_name加载
            esm_model_name: ESM模型名称或路径
            latent_dim: 潜在空间维度
            hidden_dim: 隐藏层维度
        """
        super(ESMVAE, self).__init__()
        
        # 加载或使用预加载的ESM模型
        if esm_model is None:
            self.esm = AutoModelForMaskedLM.from_pretrained(esm_model_name)
        else:
            self.esm = esm_model
        
        # 获取ESM的隐藏层维度
        esm_hidden_dim = self.esm.config.hidden_size
        
        # 编码器部分
        self.encoder_mean = nn.Linear(esm_hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(esm_hidden_dim, latent_dim)
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, esm_hidden_dim),
            nn.ReLU()
        )
        
        # 最终输出层
        self.output_layer = nn.Linear(esm_hidden_dim, len(ALPHABET))
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列
        
        Args:
            x: 输入序列张量
            
        Returns:
            潜在空间的均值和方差
        """
        # 使用ESM获取序列表示
        esm_outputs = self.esm(x, output_hidden_states=True)
        hidden_states = esm_outputs.hidden_states[-1]  # 使用最后一层隐藏状态
        pooled = hidden_states.mean(dim=1)  # 池化操作
        
        # 计算均值和方差
        mean = self.encoder_mean(pooled)
        logvar = self.encoder_logvar(pooled)
        return mean, logvar
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mean: 均值
            logvar: 对数方差
            
        Returns:
            采样得到的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量
            
        Returns:
            重建的序列logits
        """
        h = self.decoder(z)
        logits = self.output_layer(h)
        return logits
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入序列张量
            
        Returns:
            重建序列logits、均值和方差
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE损失
    
    Args:
        recon_x: 重建序列logits
        x: 原始序列
        mean: 潜在空间均值
        logvar: 潜在空间对数方差
        kl_weight: KL散度权重
        
    Returns:
        总损失、重建损失和KL散度
    """
    # 重建损失
    recon_loss = nn.CrossEntropyLoss()(recon_x, x)
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss 