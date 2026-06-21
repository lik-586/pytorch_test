# PyTorch 房价预测项目

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

基于 PyTorch 的房价预测深度学习项目，使用深度神经网络对 Kaggle GSB-544 竞赛数据集进行房价预测。

## 项目概述

本项目使用 PyTorch 构建深度神经网络模型，对爱荷华州埃姆斯市的房价进行预测。模型采用多层全连接网络架构，结合 BatchNorm 和 Dropout 技术，有效防止过拟合并加速模型收敛。

### 主要特性

- **深度神经网络架构**: 多层全连接网络 + BatchNorm + Dropout
- **自动数据预处理**: 独热编码、标准化、缺失值处理
- **训练过程可视化**: 使用 d2l.Animator 实时监控训练损失
- **Kaggle 竞赛支持**: 针对 GSB-544 Fall 2025 Regression 竞赛优化
- **梯度裁剪**: 防止梯度爆炸，提高训练稳定性
- **L2 正则化**: Adam 优化器内置权重衰减，防止过拟合

## 环境要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本

## 安装指南

### 1. 克隆仓库

```bash
git clone <repository-url>
cd pytorch_test
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据准备

项目数据位于 `gsb-544-fall-2025-regression/` 目录下：
- `train.csv`: 训练数据集
- `test.csv`: 测试数据集
- `sample_submission.csv`: 提交样例

### 运行训练

```bash
python 房价预测_GSB-544-Fall-2025-Regression.py
```

### 参数调整

在主程序中可以调整以下参数：

```python
# 在 train_and_predict() 函数中
train_and_predict(
    data_dir='./gsb-544-fall-2025-regression',
    batch_size=64,      # 批次大小
    num_epochs=200,     # 训练轮数
    lr=0.001            # 学习率
)
```

### 输出文件

- `submission.csv`: 包含预测房价的提交文件，格式为 `PID,SalePrice`

## 项目结构

```
pytorch_test/
├── .idea/                                      # PyCharm IDE 配置文件
├── gsb-544-fall-2025-regression/               # 数据目录
│   ├── sample_submission.csv                  # 提交样例
│   ├── test.csv                               # 测试数据
│   └── train.csv                               # 训练数据
├── 房价预测_GSB-544-Fall-2025-Regression.py   # 主程序
├── submission.csv                              # 预测结果
├── README.md                                   # 项目文档
└── requirements.txt                            # 依赖清单
```

## 模型架构

### 网络结构

```
DeepRegressionNet
├── 输入层 (input_dim → 256)
│   ├── Linear(input_dim, 256)
│   ├── BatchNorm1d(256)
│   ├── ReLU
│   └── Dropout(0.3)
├── 隐藏层1 (256 → 128)
│   ├── Linear(256, 128)
│   ├── BatchNorm1d(128)
│   ├── ReLU
│   └── Dropout(0.3)
├── 隐藏层2 (128 → 64)
│   ├── Linear(128, 64)
│   ├── BatchNorm1d(64)
│   ├── ReLU
│   └── Dropout(0.2)
└── 输出层 (64 → 1)
    └── Linear(64, 1)
```

### 关键技术

1. **BatchNorm**: 批量归一化，加速模型收敛，提高训练稳定性
2. **Dropout**: 随机失活，防止过拟合，提高模型泛化能力
3. **Adam 优化器**: 自适应学习率优化，比 SGD 更快更稳定
4. **梯度裁剪**: 限制梯度范数最大为 1.0，防止梯度爆炸
5. **L2 正则化**: 权重衰减系数 5e-4，防止模型过拟合

### 数据预处理流程

```
原始数据 (CSV)
    ↓
数据加载 (pandas.read_csv)
    ↓
特征分离 (train_features, test_features)
    ↓
独热编码 (pd.get_dummies)
    ↓
特征对齐 (align)
    ↓
数值标准化 (StandardScaler)
    ↓
缺失值填充 (fillna)
    ↓
张量转换 (torch.tensor)
    ↓
数据迭代器 (DataLoader)
    ↓
模型训练 (DeepRegressionNet)
    ↓
预测结果 (numpy array)
    ↓
反标准化
    ↓
结果保存 (submission.csv)
```

## 数据说明

### 数据集来源

数据集来自 Kaggle GSB-544 Fall 2025 Regression 竞赛，包含爱荷华州埃姆斯市的房屋销售数据。

### 特征说明

数据集包含以下主要特征：
- **Lot Frontage**: 临街宽度
- **Lot Area**: 地块面积
- **Neighborhood**: 社区位置
- **Overall Qual**: 整体质量评分
- **Year Built**: 建造年份
- **Gr Liv Area**: 地上居住面积
- **TotRms AbvGrd**: 地上房间总数
- 等其他特征...

### 目标变量

- **SalePrice**: 房屋销售价格（目标预测值）

## 训练监控

训练过程中会实时显示损失曲线，使用 d2l.Animator 进行可视化：

```
Epoch [10/200], Loss: 0.8234
Epoch [20/200], Loss: 0.5621
...
最终训练损失: 0.1234
```

## 贡献指南

欢迎对项目进行改进！请遵循以下步骤：

### 提交 Issue

如果您发现 bug 或有新功能建议，请：
1. 检查是否已有相关 Issue
2. 创建新 Issue，详细描述问题或建议
3. 添加适当的标签（bug、enhancement 等）

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码风格指南
- 添加适当的注释和文档字符串
- 确保代码通过所有测试

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，欢迎通过 Issue 与我们联系。

---

**注意**: 本项目仅用于学习和研究目的，不保证在实际生产环境中的性能和准确性。