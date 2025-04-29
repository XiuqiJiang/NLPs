import torch
import numpy as np
import os
import sys
import time
import logging
import traceback
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config.config import *
from models.vae import ESMVAE

def setup_logging():
    """设置日志记录"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        LOG_DIR,
        f'generate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("日志系统初始化完成")

def log_system_info():
    """记录系统信息"""
    logging.info("系统信息:")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"PyTorch版本: {torch.__version__}")
    logging.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA版本: {torch.version.cuda}")
        logging.info(f"当前GPU: {torch.cuda.get_device_name(0)}")

def log_config_info():
    """记录配置信息"""
    logging.info("当前配置:")
    logging.info(f"NUM_SAMPLES: {NUM_SAMPLES}")
    logging.info(f"LATENT_DIM: {LATENT_DIM}")
    logging.info(f"MAX_SEQUENCE_LENGTH: {MAX_SEQUENCE_LENGTH}")
    logging.info(f"MODEL_WEIGHTS_PATH: {MODEL_WEIGHTS_PATH}")
    logging.info(f"GENERATED_SEQUENCES_PATH: {GENERATED_SEQUENCES_PATH}")
    logging.info(f"ESM_MODEL_NAME: {ESM_MODEL_NAME}")

def evaluate_sequences(sequences):
    """评估生成的序列"""
    metrics = {
        'num_sequences': len(sequences),
        'avg_length': np.mean([len(seq) for seq in sequences]),
        'unique_sequences': len(set(sequences))
    }
    return metrics

def generate_sequences(
    model,
    num_samples=NUM_SAMPLES,
    latent_dim=LATENT_DIM,
    temperature=1.0
):
    """生成新的序列
    
    Args:
        model: VAE模型
        num_samples: 生成样本数
        latent_dim: 潜在空间维度
        temperature: 采样温度，控制生成的多样性
    """
    try:
        if not isinstance(model, ESMVAE):
            raise ValueError("模型必须是ESMVAE类型")
        if num_samples <= 0:
            raise ValueError("生成样本数必须大于0")
        
        device = next(model.parameters()).device
        model.eval()
        
        logging.info(f"开始生成序列，使用设备: {device}")
        logging.info(f"生成参数: num_samples={num_samples}, temperature={temperature}")
        
        generated_sequences = []
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(num_samples):
                batch_start_time = time.time()
                
                # 从标准正态分布采样
                z = torch.randn(1, latent_dim).to(device)
                
                # 解码生成序列
                logits = model.decode(z)
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                tokens = torch.argmax(probs, dim=-1)
                
                # 将tokens转换为序列
                sequence = ''.join([ALPHABET[token] for token in tokens[0]])
                generated_sequences.append(sequence)
                
                # 记录每个序列的生成信息
                batch_time = time.time() - batch_start_time
                logging.info(f"生成第 {i+1}/{num_samples} 个序列:")
                logging.info(f"序列长度: {len(sequence)}")
                logging.info(f"生成耗时: {batch_time:.2f}秒")
                if torch.cuda.is_available():
                    logging.info(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        # 记录总体生成信息
        total_time = time.time() - start_time
        metrics = evaluate_sequences(generated_sequences)
        logging.info("生成完成:")
        logging.info(f"总耗时: {total_time:.2f}秒")
        logging.info(f"平均每个序列耗时: {total_time/num_samples:.2f}秒")
        logging.info(f"生成序列总数: {metrics['num_sequences']}")
        logging.info(f"平均序列长度: {metrics['avg_length']:.2f}")
        logging.info(f"唯一序列数: {metrics['unique_sequences']}")
        
        return generated_sequences
        
    except Exception as e:
        logging.error(f"生成序列时发生错误: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误位置: {traceback.format_exc()}")
        raise

def main():
    try:
        # 设置日志
        setup_logging()
        logging.info("开始序列生成任务")
        
        # 记录系统信息
        log_system_info()
        
        # 记录配置信息
        log_config_info()
        
        # 设置随机种子
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        logging.info(f"设置随机种子: {RANDOM_SEED}")
        
        # 检查ESM模型路径
        if not os.path.exists(ESM_MODEL_NAME):
            raise FileNotFoundError(f"找不到ESM模型: {ESM_MODEL_NAME}")
        
        # 加载本地ESM模型和tokenizer
        logging.info(f"加载本地ESM模型: {ESM_MODEL_NAME}")
        esm_model = AutoModelForMaskedLM.from_pretrained(ESM_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        
        # 初始化VAE模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        model = ESMVAE(esm_model=esm_model).to(device)
        
        # 检查并加载模型权重
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            logging.warning(f"找不到模型权重文件: {MODEL_WEIGHTS_PATH}")
            logging.info("使用随机初始化的权重")
        else:
            logging.info(f"加载模型权重: {MODEL_WEIGHTS_PATH}")
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
        
        # 生成序列
        sequences = generate_sequences(
            model,
            num_samples=NUM_SAMPLES,
            latent_dim=LATENT_DIM,
            temperature=1.0
        )
        
        # 保存生成的序列
        os.makedirs(os.path.dirname(GENERATED_SEQUENCES_PATH), exist_ok=True)
        logging.info(f"保存生成的序列到: {GENERATED_SEQUENCES_PATH}")
        with open(GENERATED_SEQUENCES_PATH, 'w') as f:
            for seq in sequences:
                f.write(f'{seq}\n')
        
        logging.info(f"成功生成 {len(sequences)} 个序列，保存至 {GENERATED_SEQUENCES_PATH}")
        logging.info("序列生成任务完成")
        
    except Exception as e:
        logging.error(f"序列生成任务失败: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"错误位置: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    main() 