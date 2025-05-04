import os
import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.models.vae_token import ESMVAEToken
from config.config import (
    ESM_MODEL_PATH,
    MODEL_SAVE_DIR,
    DEVICE,
    MAX_SEQUENCE_LENGTH,
    GENERATED_SEQUENCES_PATH
)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_path: str, device: str) -> tuple[ESMVAEToken, AutoTokenizer]:
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        (模型, tokenizer)
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 加载tokenizer以获取词汇表大小
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_PATH)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    
    # 初始化模型
    model = ESMVAEToken(
        input_dim=640,  # ESM-2的嵌入维度
        hidden_dims=[512, 256],
        latent_dim=256,
        vocab_size=vocab_size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        pad_token_id=pad_token_id,
        use_layer_norm=True,
        dropout=0.1
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_sequences(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int = 10,
    device: str = DEVICE,
    save_path: str = GENERATED_SEQUENCES_PATH
) -> list[str]:
    """生成新的序列并保存
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        device: 设备
        save_path: 保存路径
        
    Returns:
        生成的序列列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始生成 {num_sequences} 个新序列...")
    
    # 检查tokenizer的特殊token
    logger.info("检查tokenizer的特殊token...")
    logger.info(f"BOS Token ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS Token ID: {tokenizer.eos_token_id}")
    logger.info(f"CLS Token ID: {tokenizer.cls_token_id}")
    logger.info(f"PAD Token ID: {tokenizer.pad_token_id}")
    
    # 选择起始token
    if tokenizer.cls_token_id is not None:
        logger.info("使用CLS token作为起始token")
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.bos_token_id is not None:
        logger.info("使用BOS token作为起始token")
        start_token_id = tokenizer.bos_token_id
    else:
        logger.warning("没有找到合适的起始token，将使用PAD token")
        start_token_id = tokenizer.pad_token_id
    
    # 选择结束token
    if tokenizer.eos_token_id is not None:
        logger.info("使用EOS token作为结束token")
        end_token_id = tokenizer.eos_token_id
    else:
        logger.warning("没有找到结束token，将使用PAD token")
        end_token_id = tokenizer.pad_token_id
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 初始化生成序列
    generated_sequences = []
    current_token_ids = torch.full((num_sequences, 1), start_token_id, dtype=torch.long, device=device)
    
    # 逐步生成序列
    with torch.no_grad():
        for _ in range(MAX_SEQUENCE_LENGTH):
            # 获取当前时间步的logits
            logits = model.decode(z, current_token_ids)
            
            # 选择下一个token（贪婪解码）
            next_token_ids = torch.argmax(logits[:, -1:], dim=-1)
            
            # 将新token添加到序列中
            current_token_ids = torch.cat([current_token_ids, next_token_ids], dim=1)
            
            # 检查是否生成了结束符
            if (next_token_ids == end_token_id).any():
                break
    
    # 将生成的token ids转换回序列
    for i in range(num_sequences):
        sequence = tokenizer.decode(current_token_ids[i], skip_special_tokens=True)
        generated_sequences.append(sequence)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for seq in generated_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {save_path}")
    logger.info("序列生成完成！")
    
    return generated_sequences

def generate_with_temperature(
    model: ESMVAEToken,
    tokenizer: AutoTokenizer,
    num_sequences: int = 10,
    temperature: float = 1.0,
    device: str = DEVICE,
    save_path: str = GENERATED_SEQUENCES_PATH
) -> list[str]:
    """使用温度采样生成序列
    
    Args:
        model: VAE模型
        tokenizer: tokenizer
        num_sequences: 要生成的序列数量
        temperature: 温度参数，控制采样的随机性
        device: 设备
        save_path: 保存路径
        
    Returns:
        生成的序列列表
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始使用温度 {temperature} 生成 {num_sequences} 个新序列...")
    
    # 检查tokenizer的特殊token
    logger.info("检查tokenizer的特殊token...")
    logger.info(f"BOS Token ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS Token ID: {tokenizer.eos_token_id}")
    logger.info(f"CLS Token ID: {tokenizer.cls_token_id}")
    logger.info(f"PAD Token ID: {tokenizer.pad_token_id}")
    
    # 选择起始token
    if tokenizer.cls_token_id is not None:
        logger.info("使用CLS token作为起始token")
        start_token_id = tokenizer.cls_token_id
    elif tokenizer.bos_token_id is not None:
        logger.info("使用BOS token作为起始token")
        start_token_id = tokenizer.bos_token_id
    else:
        logger.warning("没有找到合适的起始token，将使用PAD token")
        start_token_id = tokenizer.pad_token_id
    
    # 选择结束token
    if tokenizer.eos_token_id is not None:
        logger.info("使用EOS token作为结束token")
        end_token_id = tokenizer.eos_token_id
    else:
        logger.warning("没有找到结束token，将使用PAD token")
        end_token_id = tokenizer.pad_token_id
    
    # 生成随机潜在向量
    z = torch.randn(num_sequences, model.latent_dim, device=device)
    
    # 初始化生成序列
    generated_sequences = []
    current_token_ids = torch.full((num_sequences, 1), start_token_id, dtype=torch.long, device=device)
    
    # 逐步生成序列
    with torch.no_grad():
        for _ in range(MAX_SEQUENCE_LENGTH):
            # 获取当前时间步的logits
            logits = model.decode(z, current_token_ids)
            
            # 应用温度采样
            logits = logits[:, -1:] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # 为每个序列采样下一个token
            next_token_ids = torch.multinomial(probs.squeeze(1), 1)
            
            # 将新token添加到序列中
            current_token_ids = torch.cat([current_token_ids, next_token_ids], dim=1)
            
            # 检查是否生成了结束符
            if (next_token_ids == end_token_id).any():
                break
    
    # 将生成的token ids转换回序列
    for i in range(num_sequences):
        sequence = tokenizer.decode(current_token_ids[i], skip_special_tokens=True)
        generated_sequences.append(sequence)
        logger.info(f"\n生成的序列 {i+1}:")
        logger.info(sequence)
    
    # 保存生成的序列
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for seq in generated_sequences:
            f.write(seq + '\n')
    
    logger.info(f"序列已保存到 {save_path}")
    logger.info("序列生成完成！")
    
    return generated_sequences

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始生成序列...")
    
    # 加载模型
    model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    logger.info(f"加载模型: {model_path}")
    model, tokenizer = load_model(model_path, DEVICE)
    
    # 生成序列
    logger.info("使用标准生成模式...")
    standard_sequences = generate_sequences(
        model=model,
        tokenizer=tokenizer,
        num_sequences=5
    )
    
    # 使用不同温度生成序列
    logger.info("\n使用温度采样生成序列...")
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        logger.info(f"\n使用温度 {temp} 生成序列:")
        temp_sequences = generate_with_temperature(
            model=model,
            tokenizer=tokenizer,
            num_sequences=3,
            temperature=temp
        )
    
    logger.info("所有序列生成完成！")

if __name__ == "__main__":
    main() 