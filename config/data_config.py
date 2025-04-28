# 数据参数
DATA_DIR = 'data'
RAW_DATA_PATH = f'{DATA_DIR}/raw/NLP_sequences_no cesin.txt'
PROCESSED_DATA_PATH = f'{DATA_DIR}/processed'
MAX_SEQUENCE_LENGTH = 48
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

# 数据增强参数
AUGMENTATION = {
    'enabled': True,
    'methods': ['substitution', 'deletion', 'insertion'],
    'probability': 0.1
}

# 数据加载参数
NUM_WORKERS = 4
PIN_MEMORY = True 