import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import esm

# 1. 读取FASTA文件
def read_fasta(file_path, label):
    seqs = []
    for record in SeqIO.parse(file_path, "fasta"):
        seqs.append((str(record.seq), label))
    return seqs

positive_seqs = read_fasta('positive_dataset.fasta', 1)
negative_seqs = read_fasta('negative_dataset.fasta', 0)
all_data = positive_seqs + negative_seqs

# 2. 特征提取（用ESM-1b embedding）
# 你需要提前下载ESM模型权重
print("Loading ESM model...")
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# 加载你微调后的权重
checkpoint = torch.load("/content/drive/MyDrive/esm_model")
esm_model.load_state_dict(checkpoint["model"], strict=False)
esm_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)

def get_esm_embedding(sequence):
    # ESM要求输入为大写字母
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    # 取去掉首尾的平均池化
    embedding = token_representations[0, 1:len(sequence)+1].mean(0)
    return embedding

# 3. 批量提取embedding
X = []
y = []
for seq, label in all_data:
    emb = get_esm_embedding(seq)
    X.append(emb)
    y.append(label)
X = np.array(X)
y = np.array(y)

# 4. 数据划分
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 5. 模型训练
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
clf.fit(X_train, y_train)

# 6. 验证与测试
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:,1]

print("验证集评估：")
print(classification_report(y_val, y_val_pred))
print("测试集评估：")
print(classification_report(y_test, y_test_pred))
print("AUC:", roc_auc_score(y_test, y_test_prob))
print("混淆矩阵：\n", confusion_matrix(y_test, y_test_pred))

# 7. 保存模型
import joblib
joblib.dump(clf, "peptide_discriminator.pkl")
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

print("模型和特征已保存。")
