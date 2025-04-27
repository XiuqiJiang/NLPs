# ESM-VAE: 基于ESM的变分自编码器

本项目实现了一个基于ESM（Evolutionary Scale Modeling）的变分自编码器（VAE），用于生成蛋白质序列。

## 项目结构

```
NLPs/
├── config/                 # 配置文件
│   └── config.py          # 模型和训练参数配置
├── src/                   # 源代码
│   ├── models/           # 模型定义
│   │   └── vae.py       # VAE模型实现
│   ├── data/            # 数据处理
│   │   └── dataset.py   # 数据集类
│   ├── utils/           # 工具函数
│   │   └── trainer.py   # 训练器
│   ├── train.py         # 训练脚本
│   └── generate.py      # 生成脚本
├── notebooks/            # Jupyter notebooks
├── results/             # 结果输出
│   ├── models/         # 模型权重
│   ├── logs/          # 训练日志
│   └── generated/     # 生成的序列
└── data/               # 数据
    └── raw/           # 原始数据
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA 11.0+ (如果使用GPU)

## 安装依赖

```bash
pip install torch transformers tqdm numpy
```

## 使用方法

1. 训练模型：
```bash
python src/train.py
```

2. 生成序列：
```bash
python src/generate.py
```

## 配置说明

主要配置参数在 `config/config.py` 中：

- 数据参数：序列长度、字母表等
- ESM模型参数：模型名称、输出目录等
- VAE模型参数：潜在维度、隐藏层维度等
- 训练参数：批次大小、学习率等
- 生成参数：生成序列数量等

## 模型架构

- 编码器：使用预训练的ESM模型
- 解码器：多层感知机
- 损失函数：重建损失 + KL散度

## 注意事项

- 确保有足够的GPU内存
- 训练前检查数据格式
- 定期保存模型检查点 