import torch
import os

# 基础配置
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据配置
DATA_PATH = 'data/raw/NLP_sequences.txt'  # 修正数据文件路径
MAX_SEQUENCE_LENGTH = 64
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
TRAIN_TEST_SPLIT = 0.15
NUM_WORKERS = 4  # 数据加载的工作进程数
PIN_MEMORY = True  # 是否将数据固定在内存中，可以加快GPU传输速度

# ESM模型配置
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
ESM_OUTPUT_DIR = "results/models/esm_finetuned"
ESM_FINETUNE_EPOCHS = 6
BATCH_SIZE = 16  # 统一使用BATCH_SIZE
ESM_LEARNING_RATE = 1e-5

# VAE模型配置
LATENT_DIM = 64
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
KL_WEIGHT = 0.1

# 训练配置
SAVE_EVERY = 5
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIPPING = 1.0
METRICS = ['loss', 'recon_loss', 'kl_loss']

# 模型保存配置
SAVE_DIR = 'results/models/vae/weights'
MODEL_SAVE_FREQUENCY = 5

# 生成配置
NUM_SAMPLES = 10
GENERATED_SEQUENCES_PATH = 'results/generated/sequences.txt'

# 日志配置
LOG_DIR = 'results/logs'
LOG_LEVEL = 'INFO'
PROGRESS_BAR = True

# 创建必要的目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GENERATED_SEQUENCES_PATH), exist_ok=True) 