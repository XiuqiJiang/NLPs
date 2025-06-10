# ESM (Evolutionary Scale Modeling) 蛋白质语言模型

本项目基于ESM2架构训练了一个蛋白质语言模型，用于蛋白质序列的表示学习和下游任务。

## 模型信息

### 基础架构
- 基础模型：ESM2 (facebook/esm2_t30_150M_UR50D)
- 模型类型：EsmForMaskedLM
- 参数量：150M

### 模型配置
- 隐藏层维度：640
- 注意力头数：20
- 隐藏层数：30
- 中间层维度：2560
- 最大序列长度：1026
- 词汇表大小：33
- 激活函数：GELU
- 位置编码：Rotary Position Embedding

## 环境要求

```bash
pip install -r requirements.txt
```

## 使用方法

### 加载模型
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_path = "esm_model"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### 序列编码
```python
sequence = "MLLAVLYCLVNGSLALG"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
```

### 掩码预测
```python
sequence = "MLLAVLYCLVNGSLALG"
masked_sequence = sequence[:5] + "<mask>" + sequence[6:]
inputs = tokenizer(masked_sequence, return_tensors="pt")
outputs = model(**inputs)
```

## 模型文件说明
- `model.safetensors`: 模型权重文件
- `config.json`: 模型配置文件
- `vocab.txt`: 词汇表文件
- `tokenizer_config.json`: 分词器配置文件
- `training_args.bin`: 训练参数文件
- `special_tokens_map.json`: 特殊token映射文件

## 注意事项
1. 模型使用float32精度
2. 支持token dropout
3. 使用rotary position embedding进行位置编码
4. 默认使用缓存机制加速推理

## 引用
如果您使用了这个模型，请引用原始ESM论文：
```
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
