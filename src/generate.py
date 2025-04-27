import torch
import numpy as np
from transformers import AutoTokenizer
from config.config import *
from models.vae import ESMVAE

def generate_sequences(model, num_samples=NUM_SAMPLES, latent_dim=LATENT_DIM):
    """生成新的序列"""
    device = next(model.parameters()).device
    model.eval()
    
    generated_sequences = []
    with torch.no_grad():
        for _ in range(num_samples):
            # 从标准正态分布采样
            z = torch.randn(1, latent_dim).to(device)
            
            # 解码生成序列
            logits = model.decode(z)
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.argmax(probs, dim=-1)
            
            # 将tokens转换为序列
            sequence = ''.join([ALPHABET[token] for token in tokens[0]])
            generated_sequences.append(sequence)
    
    return generated_sequences

def main():
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESMVAE().to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    
    # 生成序列
    sequences = generate_sequences(model)
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(GENERATED_SEQUENCES_PATH), exist_ok=True)
    with open(GENERATED_SEQUENCES_PATH, 'w') as f:
        for seq in sequences:
            f.write(f'{seq}\n')
    
    print(f'Generated {len(sequences)} sequences and saved to {GENERATED_SEQUENCES_PATH}')

if __name__ == '__main__':
    main() 