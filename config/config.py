# 数据参数
DATA_PATH = 'data/raw/NLP_sequences_no cesin.txt'
MAX_SEQUENCE_LENGTH = 64
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# ESM模型参数
ESM_MODEL_NAME = "esm_model"  # 相对路径
ESM_OUTPUT_DIR = "results/models/esm_finetuned"
ESM_FINETUNE_EPOCHS = 6
ESM_BATCH_SIZE = 16
ESM_LEARNING_RATE = 1e-5

# VAE模型参数
LATENT_DIM = 50
HIDDEN_DIM = 256
VAE_BATCH_SIZE = 32
VAE_LEARNING_RATE = 0.001
VAE_EPOCHS = 100
KL_WEIGHT = 0.1  # KL散度的权重

# 训练参数
TRAIN_TEST_SPLIT = 0.15
RANDOM_SEED = 42

# 生成参数
NUM_SAMPLES = 10
MODEL_WEIGHTS_PATH = 'results/models/vae/weights/epoch_100.pth'
GENERATED_SEQUENCES_PATH = 'results/generated/sequences.txt' 