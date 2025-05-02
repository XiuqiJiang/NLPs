import torch
import os
import logging

# 基础配置
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据配置
DATA_PATH = 'data/raw/sequences.txt'  # 原始序列数据路径
MAX_SEQUENCE_LENGTH = 64  # 最大序列长度（包括特殊token）

# 词汇表配置
ALPHABET = ['<pad>', '<eos>'] + list('ACDEFGHIKLMNPQRSTVWY')  # 特殊token + 标准氨基酸
VOCAB_SIZE = len(ALPHABET)  # 词汇表大小（包括特殊token）

# 特殊token配置
PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'
PAD_TOKEN_ID = ALPHABET.index(PAD_TOKEN)  # 0
EOS_TOKEN_ID = ALPHABET.index(EOS_TOKEN)  # 1

# 数据加载配置
TRAIN_TEST_SPLIT = 0.15
NUM_WORKERS = 4
PIN_MEMORY = True

# ESM模型配置
ESM_MODEL_PATH = "/esm_model"  # 微调后的ESM模型路径
ESM_MODEL_NAME = "facebook/esm2-t6-8M-UR50D"  # 基础模型名称（仅用于参考）
ESM_EMBEDDING_DIM = 320  # ESM-2 嵌入维度
ESM_OUTPUT_DIR = "results/models/esm_finetuned"
ESM_FINETUNE_EPOCHS = 6
BATCH_SIZE = 32
ESM_LEARNING_RATE = 1e-5

# VAE模型配置
LATENT_DIM = 256  # 潜在空间维度
HIDDEN_DIMS = [512, 256]  # 隐藏层维度
LEARNING_RATE = 1e-4  # 学习率
NUM_EPOCHS = 100
BETA = 1.0  # KL散度权重（用于调整KL散度损失的强度）
KL_WEIGHT = BETA  # 为了向后兼容，保持KL_WEIGHT

# 训练配置
SAVE_EVERY = 5  # 每多少轮保存一次模型
EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 0.01  # 早停最小变化值
GRADIENT_CLIPPING = 1.0
METRICS = ['loss', 'recon_loss', 'kl_loss']

# 模型保存配置
MODEL_SAVE_DIR = "results/models/vae/weights"  # 模型保存目录
MODEL_SAVE_FREQUENCY = 5

# 生成配置
NUM_SAMPLES = 10
GENERATED_SEQUENCES_PATH = 'results/generated/sequences.txt'

# 日志配置
LOG_DIR = "results/logs"  # 日志目录
LOG_LEVEL = logging.INFO  # 日志级别
PROGRESS_BAR = True

# 创建必要的目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GENERATED_SEQUENCES_PATH), exist_ok=True)

# 验证配置
def validate_config():
    """验证配置参数的一致性"""
    # 验证特殊token
    assert PAD_TOKEN in ALPHABET, f"PAD_TOKEN '{PAD_TOKEN}' 不在 ALPHABET 中"
    assert EOS_TOKEN in ALPHABET, f"EOS_TOKEN '{EOS_TOKEN}' 不在 ALPHABET 中"
    
    # 验证token ID
    assert PAD_TOKEN_ID == ALPHABET.index(PAD_TOKEN), f"PAD_TOKEN_ID ({PAD_TOKEN_ID}) 与 ALPHABET 中的索引不匹配"
    assert EOS_TOKEN_ID == ALPHABET.index(EOS_TOKEN), f"EOS_TOKEN_ID ({EOS_TOKEN_ID}) 与 ALPHABET 中的索引不匹配"
    
    # 验证词汇表大小
    assert VOCAB_SIZE == len(ALPHABET), f"VOCAB_SIZE ({VOCAB_SIZE}) 与 ALPHABET 长度 ({len(ALPHABET)}) 不匹配"
    
    # 验证序列长度
    assert MAX_SEQUENCE_LENGTH > 0, "MAX_SEQUENCE_LENGTH 必须大于 0"
    
    # 验证学习率
    assert LEARNING_RATE > 0, "LEARNING_RATE 必须大于 0"
    assert ESM_LEARNING_RATE > 0, "ESM_LEARNING_RATE 必须大于 0"
    
    # 验证KL权重
    assert BETA > 0, "BETA 必须大于 0"
    assert KL_WEIGHT == BETA, "KL_WEIGHT 必须等于 BETA"

# 运行配置验证
validate_config() 