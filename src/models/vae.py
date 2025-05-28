class ProteinVAE(nn.Module):
    """蛋白质序列的条件变分自编码器"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_rings: int,
        max_sequence_length: int,
        dropout: float = 0.1
    ):
        """初始化模型
        
        Args:
            input_dim: 输入维度（ESM嵌入维度）
            hidden_dim: 隐藏层维度
            latent_dim: 潜在空间维度
            num_rings: 环数类别数
            max_sequence_length: 最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_rings = num_rings
        self.max_sequence_length = max_sequence_length
        
        # 环数嵌入层
        self.ring_embedding = nn.Embedding(num_rings, hidden_dim)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 均值和对数方差预测
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor, ring_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            ring_info: 环数信息 [batch_size]
            
        Returns:
            (mu, log_var): 均值和方差
        """
        # 获取环数嵌入 [batch_size, hidden_dim]
        ring_emb = self.ring_embedding(ring_info)
        
        # 扩展环数嵌入以匹配序列长度 [batch_size, seq_len, hidden_dim]
        ring_emb = ring_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # 连接输入和环数嵌入
        x = torch.cat([x, ring_emb], dim=-1)
        
        # 编码
        h = self.encoder(x)
        
        # 预测均值和方差
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: 均值
            log_var: 对数方差
            
        Returns:
            采样的潜在向量
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, ring_info: torch.Tensor) -> torch.Tensor:
        """解码潜在向量
        
        Args:
            z: 潜在向量 [batch_size, seq_len, latent_dim]
            ring_info: 环数信息 [batch_size]
            
        Returns:
            重构的序列
        """
        # 获取环数嵌入
        ring_emb = self.ring_embedding(ring_info)
        ring_emb = ring_emb.unsqueeze(1).expand(-1, z.size(1), -1)
        
        # 连接潜在向量和环数嵌入
        z = torch.cat([z, ring_emb], dim=-1)
        
        # 解码
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        ring_info: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_dim]
            ring_info: 环数信息 [batch_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            包含以下键的字典:
            - 'recon_x': 重构的序列
            - 'mu': 均值
            - 'log_var': 对数方差
            - 'z': 潜在向量
        """
        # 编码
        mu, log_var = self.encode(x, ring_info)
        
        # 重参数化
        z = self.reparameterize(mu, log_var)
        
        # 解码
        recon_x = self.decode(z, ring_info)
        
        # 如果提供了注意力掩码，将填充位置的重构输出置为0
        if attention_mask is not None:
            recon_x = recon_x * attention_mask.unsqueeze(-1)
        
        return {
            'recon_x': recon_x,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def generate(
        self,
        ring_info: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """生成序列
        
        Args:
            ring_info: 环数信息 [batch_size]
            num_samples: 每个环数生成的样本数
            temperature: 采样温度
            
        Returns:
            生成的序列 [batch_size * num_samples, seq_len, input_dim]
        """
        self.eval()
        with torch.no_grad():
            # 扩展环数信息
            ring_info = ring_info.repeat_interleave(num_samples, dim=0)
            
            # 从标准正态分布采样
            z = torch.randn(
                len(ring_info),
                self.max_sequence_length,
                self.latent_dim,
                device=ring_info.device
            ) * temperature
            
            # 解码
            return self.decode(z, ring_info) 