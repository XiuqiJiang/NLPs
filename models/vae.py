import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from config import *

class ESMVAE(nn.Module):
    def __init__(self, esm_model_name=ESM_MODEL_NAME, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super(ESMVAE, self).__init__()
        
        # 加载预训练的ESM模型作为编码器
        self.esm = AutoModelForMaskedLM.from_pretrained(esm_model_name)
        
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
        
    def encode(self, x):
        # 使用ESM获取序列表示
        esm_outputs = self.esm(x, output_hidden_states=True)
        hidden_states = esm_outputs.hidden_states[-1]  # 使用最后一层隐藏状态
        pooled = hidden_states.mean(dim=1)  # 池化操作
        
        # 计算均值和方差
        mean = self.encoder_mean(pooled)
        logvar = self.encoder_logvar(pooled)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        logits = self.output_layer(h)
        return logits
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def vae_loss(recon_x, x, mean, logvar, kl_weight=KL_WEIGHT):
    # 重建损失
    recon_loss = nn.CrossEntropyLoss()(recon_x, x)
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss 