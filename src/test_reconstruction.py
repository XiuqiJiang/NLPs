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

        # scheduled_sampling_prob对比
        for ss_prob in [0.0, 0.1, 0.3, 0.5]:
            logits = model.decode(
                model.encode(embeddings, attention_mask)[0],
                target_ids=input_ids,
                scheduled_sampling_prob=ss_prob
            )
            pred_ids_tf = logits.argmax(dim=-1)
            for j in range(embeddings.size(0)):
                input_token_ids = input_ids[j].cpu().tolist()
                pred_token_ids_tf = pred_ids_tf[j].cpu().tolist()
                input_tokens = tokenizer.decode(input_token_ids, skip_special_tokens=True)
                pred_tokens_tf = tokenizer.decode(pred_token_ids_tf, skip_special_tokens=True)
                def simple_edit_distance(a, b):
                    return sum(x != y for x, y in zip(a, b))
                edit_dist_tf = simple_edit_distance(input_token_ids, pred_token_ids_tf)
                diff_pos_tf = [k for k, (x, y) in enumerate(zip(input_token_ids, pred_token_ids_tf)) if x != y]
                print(f"样本 {i * val_loader.batch_size + j} [SS={ss_prob}]:")
                print(f"原始token id: {input_token_ids}")
                print(f"Teacher Forcing重构: {pred_token_ids_tf}")
                print(f"原始序列: {input_tokens}")
                print(f"Teacher Forcing重构: {pred_tokens_tf}")
                print(f"[TF] 编辑距离: {edit_dist_tf}，差异位置: {diff_pos_tf}")
                print('-' * 40)
        # 自回归生成
        z = model.encode(embeddings, attention_mask)[0]
        pred_ids_ar = model.decode(z, target_ids=None)
        for j in range(embeddings.size(0)):
            input_token_ids = input_ids[j].cpu().tolist()
            pred_token_ids_ar = pred_ids_ar[j].cpu().tolist()
            input_tokens = tokenizer.decode(input_token_ids, skip_special_tokens=True)
            pred_tokens_ar = tokenizer.decode(pred_token_ids_ar, skip_special_tokens=True)
            def simple_edit_distance(a, b):
                return sum(x != y for x, y in zip(a, b))
            edit_dist_ar = simple_edit_distance(input_token_ids, pred_token_ids_ar)
            diff_pos_ar = [k for k, (x, y) in enumerate(zip(input_token_ids, pred_token_ids_ar)) if x != y]
            print(f"样本 {i * val_loader.batch_size + j} [AR]:")
            print(f"原始token id: {input_token_ids}")
            print(f"自回归生成: {pred_token_ids_ar}")
            print(f"原始序列: {input_tokens}")
            print(f"自回归生成序列: {pred_tokens_ar}")
            print(f"[AR] 编辑距离: {edit_dist_ar}，差异位置: {diff_pos_ar}")
            print('-' * 40)
        if i >= 9:
            break