import torch
from transformers import AutoTokenizer
from src.models.vae_token import ESMVAEToken
from src.utils.data_utils import create_data_loaders
from config.config import (
    ESM_MODEL_PATH,
    HIDDEN_DIMS,
    LATENT_DIM,
    RNN_HIDDEN_DIM,
    MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_DIR
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    ESM_MODEL_PATH,
    local_files_only=True
)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ESMVAEToken(
    input_dim=1280,  # 根据你的实际embedding维度
    hidden_dims=HIDDEN_DIMS,
    latent_dim=LATENT_DIM,
    vocab_size=tokenizer.vocab_size,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    pad_token_id=tokenizer.pad_token_id,
    use_layer_norm=True,
    dropout=0.1,
    num_classes=3,
    rnn_hidden_dim=RNN_HIDDEN_DIM
).to(device)

# 加载最佳模型权重
checkpoint = torch.load(f"{MODEL_SAVE_DIR}/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载数据
_, val_loader = create_data_loaders(
    data_file='data/processed/protein_data.pt',  # 使用项目默认数据文件路径
    batch_size=1,
    train_test_split=0.1,
    num_workers=0,
    pin_memory=False,
    max_sequence_length=MAX_SEQUENCE_LENGTH
)

# 测试重构
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        embeddings = batch['embeddings'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ring_info = batch['ring_info'].to(device)

        # 前向传播，获取重构logits
        logits, _, _, _, _ = model(embeddings, attention_mask, input_ids, ring_info)
        pred_ids = logits.argmax(dim=-1)  # [batch, seq_len]

        # 解码为字符串
        for j in range(embeddings.size(0)):
            input_tokens = tokenizer.decode(input_ids[j].cpu().numpy(), skip_special_tokens=True)
            pred_tokens = tokenizer.decode(pred_ids[j].cpu().numpy(), skip_special_tokens=True)
            print(f"样本 {i * val_loader.batch_size + j}:")
            print(f"原始序列: {input_tokens}")
            print(f"重构序列: {pred_tokens}")
            print('-' * 40)

        if i >= 9:  # 只看前10个样本
            break