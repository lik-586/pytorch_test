import torch
from torch import nn
import pandas as pd
from d2l import torch as d2l

# 1. 加载和预处理数据
def load_data(data_dir):
    # 读取数据
    train_data = pd.read_csv(f'{data_dir}/train.csv')
    test_data = pd.read_csv(f'{data_dir}/test.csv')

    # 分离特征和标签
    train_features = train_data.iloc[:, 1:-1]  # 除去PID和SalePrice
    test_features = test_data.iloc[:, 1:]      # 除去PID

    # 处理类别特征：独热编码（先对所有特征进行独热编码，再分离数值和类别）
    # pd.get_dummies 会自动将字符串列转换为独热向量
    train_features = pd.get_dummies(train_features, dummy_na=True)
    test_features = pd.get_dummies(test_features, dummy_na=True)

    # 确保训练集和测试集的特征列一致（对齐）
    train_features, test_features = train_features.align(test_features, join='inner', axis=1)

    # 处理数值特征：标准化（在对齐后处理，确保列顺序一致）
    # 计算训练集的均值和标准差，用于标准化测试集
    numeric_features = train_features.dtypes[train_features.dtypes != 'object'].index
    train_features[numeric_features] = train_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # 填充缺失值为0（标准化后均值为0）
    train_features[numeric_features] = train_features[numeric_features].fillna(0)
    test_features[numeric_features] = test_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    test_features[numeric_features] = test_features[numeric_features].fillna(0)

    # 计算 y_train 的均值和标准差
    y_mean = train_data['SalePrice'].mean()
    y_std = train_data['SalePrice'].std()
    # 保存用于后续反标准化（预测时还原）

    # 转换为张量
    X_train = torch.tensor(train_features.values, dtype=torch.float32)
    y_train = torch.tensor((train_data['SalePrice'].values - y_mean) / y_std, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(test_features.values, dtype=torch.float32)

    return X_train, y_train, X_test, y_mean, y_std

# 2. 定义网络结构
# 设计一个包含 BatchNorm 和 Dropout 的深层网络
class DeepRegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(DeepRegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),  # 批量归一化，加速收敛
            nn.ReLU(),
            nn.Dropout(0.3),      # Dropout层，防止过拟合

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)     # 回归任务，输出1个数值
        )

    def forward(self, x):
        return self.net(x)

# 3. 训练与评估
def train_and_predict(data_dir, batch_size=64, num_epochs=100, lr=0.001):
    # 加载数据
    X_train, y_train, X_test, y_mean, y_std = load_data(data_dir)

    # 定义数据迭代器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    # 初始化网络
    input_dim = X_train.shape[1]
    net = DeepRegressionNet(input_dim)

    # 定义损失函数和优化器
    # 为了数值稳定性，使用MSELoss
    loss = nn.MSELoss()

    # 使用 Adam 优化器，比 SGD 更快更稳定
    trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

    # 训练过程
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], legend=['train'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 添加梯度裁剪，防止梯度爆炸
            # max_norm=1 表示梯度的范数最大为 1，超过的部分会被缩放
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            trainer.step()

        # 记录训练损失
        if (epoch + 1) % 10 == 0:
            train_l = loss(net(X_train), y_train)
            animator.add(epoch + 1, (train_l.detach().cpu().numpy(),))

    print(f'最终训练损失: {train_l.item()}')

    # 预测
    net.eval()
    preds = net(X_test).detach().numpy()
    # 【修正】反标准化：将预测结果还原为原始房价范围
    preds = preds * y_std + y_mean

    # 保存预测结果到 CSV (用于提交Kaggle)
    test_data = pd.read_csv(f'{data_dir}/test.csv')
    sub_data = pd.DataFrame({'PID': test_data['PID'], 'SalePrice': preds.reshape(1, -1)[0]})
    sub_data.to_csv('submission.csv', index=False)

    return preds

# 主程序
if __name__ == '__main__':
    # 数据路径
    DATA_DIR = './gsb-544-fall-2025-regression'

    # 运行训练和预测
    predictions = train_and_predict(DATA_DIR, num_epochs=200)