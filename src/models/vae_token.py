import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Tuple, Optional, List, Dict
from config.config import (
    ESM_MODEL_NAME,
    LATENT_DIM,
    HIDDEN_DIMS,
    ESM_EMBEDDING_DIM,
    get_beta,
    MAX_BETA,
    KLD_TARGET,
    KLD_TARGET_WEIGHT,
    BETA_FREEZE_EPOCHS
)

RING_LOSS_WEIGHT = 0  # 只训练重构，不加环数损失

class ESMVAEToken(nn.Module):
    """基于ESM嵌入的极简VAE模型，输出token序列"""
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        hidden_dims: list = [1024, 512],
        latent_dim: int = 512,
        use_layer_norm: bool = True,
        dropout: float = 0.2,
        rnn_hidden_dim: int = 512,
        num_rnn_layers: int = 1
    ):
        """初始化VAE模型
        
        Args:
            input_dim: 输入维度（ESM嵌入维度）
            vocab_size: 词汇表大小
            max_sequence_length: 最大序列长度
            pad_token_id: padding token的ID
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            use_layer_norm: 是否使用LayerNorm
            dropout: dropout比率
            rnn_hidden_dim: RNN隐藏层维度
            num_rnn_layers: RNN层数
        """
        super().__init__()
        self.dropout = dropout
        self.token_dropout_rate = 0.0
        self.unk_token_id = getattr(self, 'unk_token_id', 3)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.use_layer_norm = use_layer_norm
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers
        # 编码器
        self.encoder = self.build_encoder()
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        # Transformer解码器替换RNN
        self.decoder_embedding = nn.Embedding(vocab_size, rnn_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=rnn_hidden_dim,
            nhead=8,
            dim_feedforward=4*rnn_hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.latent_to_memory = nn.Linear(latent_dim, rnn_hidden_dim)
        self.fc_out = nn.Linear(rnn_hidden_dim, vocab_size)
        # 位置编码
        self.pos_embedding = nn.Embedding(self.max_sequence_length, self.rnn_hidden_dim)
        # 特殊token IDs
        self.sos_token_id = 0
        self.eos_token_id = 2
        self.apply(self._init_weights)
        print(f"Model configuration:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Hidden dimensions: {hidden_dims}")
        print(f"- Latent dimension: {latent_dim}")
        print(f"- Vocabulary size: {vocab_size}")
        print(f"- RNN hidden dimension: {rnn_hidden_dim}")
    
    def build_encoder(self) -> nn.Module:
        """构建编码器网络"""
        layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if self.use_layer_norm else nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.0)
            ])
            in_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码输入序列
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            
        Returns:
            mu: 均值，形状为 [batch_size, latent_dim]
            logvar: 对数方差，形状为 [batch_size, latent_dim]
        """
        # 验证输入形状
        if x.dim() != 3:
            raise ValueError(f"输入张量必须是3维的 [batch_size, seq_len, input_dim]，但收到了 {x.shape}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"输入维度必须是 {self.input_dim}，但收到了 {x.size(-1)}")
        
        # 验证注意力掩码
        if attention_mask is not None:
            if attention_mask.shape != x.shape[:2]:
                raise ValueError(f"注意力掩码形状 {attention_mask.shape} 与输入形状 {x.shape[:2]} 不匹配")
            if attention_mask.dtype != torch.long:
                raise ValueError(f"注意力掩码必须是长整型，但收到了 {attention_mask.dtype}")
        
        # 使用注意力掩码进行池化
        if attention_mask is not None:
            # 将掩码扩展到与输入相同的维度
            mask = attention_mask.unsqueeze(-1).float()
            # 计算掩码下的平均值
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # 添加小量避免除零
        else:
            # 如果没有掩码，直接取平均值
            x = x.mean(dim=1)
        
        # 通过编码器网络
        x = self.encoder(x)
        
        # 计算均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            z: 采样的潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, target_ids: Optional[torch.Tensor] = None, gumbel_softmax: bool = False, scheduled_sampling_prob: float = 0.1, temperature: float = 1.0) -> torch.Tensor:
        batch_size = z.size(0)
        device = z.device
        memory = self.latent_to_memory(z).unsqueeze(1)  # [B, 1, d_model]
        max_len = self.max_sequence_length
        if target_ids is not None:
            seq_length = target_ids.size(1)
            if seq_length > max_len:
                target_ids = target_ids[:, :max_len]
                seq_length = max_len
            decoder_input_ids = target_ids[:, :-1]
            sos_tokens = torch.full((batch_size, 1), self.sos_token_id, device=device)
            decoder_input_ids = torch.cat([sos_tokens, decoder_input_ids], dim=1)
            if self.training and self.token_dropout_rate > 0:
                mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < self.token_dropout_rate
                decoder_input_ids = decoder_input_ids.clone()
                decoder_input_ids[mask] = self.unk_token_id
            # scheduled sampling（仅在训练时）
            if self.training and scheduled_sampling_prob > 0:
                decoder_input_ids_ss = decoder_input_ids.clone()
                for t in range(1, decoder_input_ids.shape[1]):
                    if torch.rand(1).item() < scheduled_sampling_prob:
                        prev_inputs = decoder_input_ids_ss[:, :t]
                        prev_emb = self.decoder_embedding(prev_inputs)
                        # 加入位置编码
                        positions = torch.arange(prev_emb.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
                        prev_emb = prev_emb + self.pos_embedding(positions)
                        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t).to(device)
                        out = self.transformer_decoder(
                            tgt=prev_emb,
                            memory=memory.expand(batch_size, -1, -1),
                            tgt_mask=tgt_mask
                        )
                        logits = self.fc_out(out[:, -1])
                        pred_token = logits.argmax(dim=-1)
                        not_pad = decoder_input_ids_ss[:, t] != self.pad_token_id
                        decoder_input_ids_ss[not_pad, t] = pred_token[not_pad]
                decoder_input_ids = decoder_input_ids_ss
            # embedding
            tgt_emb = self.decoder_embedding(decoder_input_ids)
            # 加入位置编码
            positions = torch.arange(tgt_emb.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_emb = tgt_emb + self.pos_embedding(positions)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
            out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory.expand(batch_size, -1, -1),
                tgt_mask=tgt_mask
            )
            logits = self.fc_out(out)
            # ====== DEBUG PRINT（teacher forcing重构）======
            # print("[DEBUG] target_ids[0]:", target_ids[0].detach().cpu().tolist())
            # print("[DEBUG] decoder_input_ids[0]:", decoder_input_ids[0].detach().cpu().tolist())
            # print("[DEBUG] logits.argmax(-1)[0]:", logits[0].argmax(-1).detach().cpu().tolist())
            # if batch_size > 1:
            #     print("[DEBUG] target_ids[1]:", target_ids[1].detach().cpu().tolist())
            #     print("[DEBUG] decoder_input_ids[1]:", decoder_input_ids[1].detach().cpu().tolist())
            #     print("[DEBUG] logits.argmax(-1)[1]:", logits[1].argmax(-1).detach().cpu().tolist())
            # ====== END DEBUG PRINT ======
            return logits
        else:
            # 自回归生成（softmax采样）
            generated = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for t in range(max_len):
                tgt_emb = self.decoder_embedding(generated)
                positions = torch.arange(tgt_emb.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
                tgt_emb = tgt_emb + self.pos_embedding(positions)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
                out = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=memory.expand(batch_size, -1, -1),
                    tgt_mask=tgt_mask
                )
                logits = self.fc_out(out[:, -1])  # [B, vocab_size]
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                generated = torch.cat([generated, next_token], dim=1)
                finished = finished | (next_token.squeeze(1) == self.eos_token_id) | (next_token.squeeze(1) == self.pad_token_id)
                if finished.all():
                    break
            return generated[:, 1:]
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        scheduled_sampling_prob: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            target_ids: 目标token IDs，形状为 [batch_size, max_sequence_length]
            scheduled_sampling_prob: scheduled sampling概率
            
        Returns:
            (重构logits/生成token IDs, 输入token IDs, 均值, 对数方差)
        """
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        if target_ids is not None:
            logits = self.decode(z, target_ids, scheduled_sampling_prob=scheduled_sampling_prob)
            return logits, target_ids, mu, logvar
        else:
            generated_ids = self.decode(z, None)
            return generated_ids, None, mu, logvar

def vae_token_loss(
    recon_logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    input_ids: torch.Tensor,     # [batch_size, seq_len]
    mu: torch.Tensor,            # [batch_size, latent_dim]
    logvar: torch.Tensor,        # [batch_size, latent_dim]
    epoch: int,
    pad_token_id: int = 1,
    beta: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    device = recon_logits.device
    input_ids = input_ids.to(device)
    mu = mu.to(device)
    logvar = logvar.to(device)
    # 只在非pad位置计算loss
    active_loss = input_ids != pad_token_id
    recon_loss = F.cross_entropy(
        recon_logits.view(-1, recon_logits.size(-1)),
        input_ids.view(-1),
        ignore_index=pad_token_id
    )
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_element = kld_element.mean(dim=0)
    free_bits = 0.02
    kld_element = torch.clamp(kld_element, min=free_bits)
    kld_loss = kld_element.sum()
    kld_raw_mean = kld_loss.detach().item() if isinstance(kld_loss, torch.Tensor) else float(kld_loss)
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, beta, kld_loss, kld_raw_mean 