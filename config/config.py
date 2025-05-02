import torch
import os
import logging

# 基础配置
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据配置
DATA_PATH = 'data/raw/sequences.txt'  # 修正数据文件路径
MAX_SEQUENCE_LENGTH = 64
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'  # 标准氨基酸
SPECIAL_TOKENS = ['<pad>', '<eos>']  # 特殊token
VOCAB_SIZE = len(ALPHABET) + 2  # 词汇表大小（包括特殊token）
PAD_TOKEN_ID = 0  # padding token的ID
EOS_TOKEN_ID = 1  # end of sequence token的ID
TRAIN_TEST_SPLIT = 0.15
NUM_WORKERS = 4  # 数据加载的工作进程数
PIN_MEMORY = True  # 是否将数据固定在内存中，可以加快GPU传输速度

# ESM模型配置
ESM_MODEL_PATH = "/esm_model"  # 微调后的ESM模型路径
ESM_MODEL_NAME = "facebook/esm2-t6-8M-UR50D"  # 基础模型名称（仅用于参考）
ESM_EMBEDDING_DIM = 320  # ESM-2 嵌入维度
ESM_OUTPUT_DIR = "results/models/esm_finetuned"  # ESM微调输出目录
ESM_FINETUNE_EPOCHS = 6
BATCH_SIZE = 32
ESM_LEARNING_RATE = 1e-5

# 确保ESM模型目录存在
os.makedirs(ESM_MODEL_PATH, exist_ok=True)

# VAE模型配置
LATENT_DIM = 256  # 潜在空间维度
HIDDEN_DIMS = [512, 256]  # 隐藏层维度
LEARNING_RATE = 1e-4  # 学习率
NUM_EPOCHS = 100
KL_WEIGHT = 0.1  # KL散度权重

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