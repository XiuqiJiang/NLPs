import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIM,
    ALPHABET
)

class ESMVAEToken(nn.Module):
    """基于ESM的变分自编码器，直接生成token ID"""
    
    def __init__(
        self,
        embedding_dim: int = 320,
        latent_dim: int = 64,
        hidden_dims: List[int] = [512, 256]
    ) -> None:
        """初始化VAE
        
        Args:
            embedding_dim: 输入embedding的维度
            latent_dim: 隐变量的维度
            hidden_dims: 编码器和解码器的隐藏层维度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # 加载ESM模型和tokenizer
        self.esm_model = AutoModelForMaskedLM.from_pretrained(ESM_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.vocab_size = self.tokenizer.vocab_size
        
        # 编码器
        encoder_layers = []
        in_features = embedding_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ])
            in_features = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和对数方差层
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_layers = []
        in_features = latent_dim
        
        # 第一层：从latent_dim映射到第一个隐藏层维度
        decoder_layers.extend([
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ])
        
        # 后续层
        for h_dim in reversed(hidden_dims[1:]):
            decoder_layers.extend([
                nn.Linear(hidden_dims[0], h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ])
        
        # 最后一层：映射到词汇表大小
        decoder_layers.append(nn.Linear(hidden_dims[-1], self.vocab_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 打印模型结构
        print(f"VAE模型结构:")
        print(f"隐变量维度: {latent_dim}")
        print(f"隐藏层维度: {hidden_dims}")
        print(f"输入维度: {embedding_dim}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"解码器结构: {self.decoder}")
    
    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入数据
        
        Args:
            x: 包含embeddings的字典
            
        Returns:
            均值和方差
        """
        embeddings = x['embeddings']
        h = self.encoder(embeddings)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            采样得到的隐变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码隐变量
        
        Args:
            z: 隐变量
            
        Returns:
            预测的logits
        """
        return self.decoder(z)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 包含embeddings和input_ids的字典
            
        Returns:
            预测logits、均值和方差
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

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
        recon_logits: 预测的logits
        input_ids: 真实的token ID
        mean: 潜在空间均值
        logvar: 潜在空间对数方差
        kl_weight: KL散度权重
        pad_token_id: padding token的ID
        
    Returns:
        总损失、重建损失和KL散度
    """
    # 重建损失 (使用交叉熵损失)
    recon_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1)
    )
    
    # KL散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss 