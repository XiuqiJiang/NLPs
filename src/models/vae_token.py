import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIMS,
    ALPHABET,
    ESM_EMBEDDING_DIM
)

class ESMVAEToken(nn.Module):
    """基于 ESM embeddings 的 VAE 模型，输出 token 序列"""
    
    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        vocab_size: int,
        max_sequence_length: int,
        use_layer_norm: bool = True
    ) -> None:
        """初始化 VAE 模型
        
        Args:
            embedding_dim: 输入 embedding 的维度
            latent_dim: 潜在空间的维度
            hidden_dims: 编码器和解码器的隐藏层维度列表
            vocab_size: 词汇表大小（包括特殊 token）
            max_sequence_length: 最大序列长度
            use_layer_norm: 是否使用 LayerNorm（默认为 True）
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_layer_norm = use_layer_norm
        
        # 构建编码器和解码器
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        # 编码器的最终层（均值和对数方差）
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
    
    def build_encoder(self) -> nn.Sequential:
        """构建编码器网络
        
        Returns:
            编码器网络
        """
        modules = []
        in_features = self.embedding_dim
        
        # 添加隐藏层
        for h_dim in self.hidden_dims:
            # 对每个 token 的 embedding 进行线性变换
            modules.append(nn.Linear(in_features, h_dim))
            modules.append(nn.ReLU())
            
            if self.use_layer_norm:
                # 使用 LayerNorm 处理序列数据
                modules.append(nn.LayerNorm(h_dim))
            else:
                # 使用 BatchNorm1d，需要调整维度
                modules.append(nn.BatchNorm1d(h_dim))
            
            in_features = h_dim
        
        # 添加池化层，将序列信息压缩成一个向量
        modules.append(nn.Sequential(
            nn.Linear(in_features, in_features),  # 可选的全连接层
            nn.ReLU(),
            nn.LayerNorm(in_features) if self.use_layer_norm else nn.BatchNorm1d(in_features)
        ))
        
        return nn.Sequential(*modules)
    
    def build_decoder(self) -> nn.Sequential:
        """构建解码器网络
        
        Returns:
            解码器网络
        """
        modules = []
        in_features = self.latent_dim
        
        # 添加隐藏层
        for h_dim in reversed(self.hidden_dims):
            modules.append(nn.Linear(in_features, h_dim))
            modules.append(nn.ReLU())
            
            if self.use_layer_norm:
                modules.append(nn.LayerNorm(h_dim))
            else:
                modules.append(nn.BatchNorm1d(h_dim))
            
            in_features = h_dim
        
        # 添加输出层，将隐藏状态映射到 token logits
        modules.append(nn.Linear(in_features, self.max_sequence_length * self.vocab_size))
        
        return nn.Sequential(*modules)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列
        
        Args:
            x: 输入序列，形状为 [batch_size, seq_len, embedding_dim]
            
        Returns:
            均值和对数方差，形状均为 [batch_size, latent_dim]
        """
        batch_size = x.size(0)
        
        # 对每个 token 的 embedding 进行编码
        x = self.encoder(x)  # [batch_size, seq_len, hidden_dim]
        
        # 使用平均池化将序列信息压缩成一个向量
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        # 计算均值和方差
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量，形状为 [batch_size, latent_dim]
            
        Returns:
            解码后的序列 logits，形状为 [batch_size, seq_len, vocab_size]
        """
        # 通过解码器网络
        x = self.decoder(z)  # [batch_size, seq_len * vocab_size]
        
        # 重塑为序列形式
        x = x.view(-1, self.max_sequence_length, self.vocab_size)
        
        return x
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: 均值，形状为 [batch_size, latent_dim]
            log_var: 对数方差，形状为 [batch_size, latent_dim]
            
        Returns:
            采样得到的潜在向量，形状为 [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入序列，形状为 [batch_size, seq_len, embedding_dim]
            
        Returns:
            (重构序列, 均值, 对数方差)
            重构序列形状为 [batch_size, seq_len, vocab_size]
            均值和方差形状均为 [batch_size, latent_dim]
        """
        # 编码
        mu, log_var = self.encode(x)
        
        # 重参数化
        z = self.reparameterize(mu, log_var)
        
        # 解码
        x_recon = self.decode(z)
        
        return x_recon, mu, log_var

def vae_token_loss(
    recon_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.1,
    pad_token_id: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算VAE损失
    
    Args:
        recon_logits: 预测的logits，形状为(batch_size, sequence_length, vocab_size)
        input_ids: 真实的token ID，形状为(batch_size, sequence_length)
        mean: 潜在空间均值
        logvar: 潜在空间对数方差
        kl_weight: KL散度权重
        pad_token_id: padding token的ID
        
    Returns:
        总损失、重建损失和KL散度
    """
    # 重建损失 (使用交叉熵损失)
    # 将logits和targets展平为2D张量
    recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss 