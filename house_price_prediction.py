import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据集准备 ---
def load_data(batch_size=10):
    # 加载波士顿房价数据 (注意：新版 sklearn 需要从 datasets 中导入)
    boston = load_boston()
    X, y = boston.data, boston.target

    # 数据标准化 (非常重要，否则神经网络很难收敛)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    # 创建 DataLoader
    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_iter = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter, scaler_y

# --- 2. 自主设计的神经网络结构 ---
class CustomNet(nn.Module):
    def __init__(self, input_size):
        super(CustomNet, self).__init__()
        # 网络结构设计思路：
        # 1. 输入层 -> 64维隐藏层 (使用 ReLU)
        # 2. 64维 -> 32维隐藏层 (使用 ReLU)
        # 3. 32维 -> 1维输出层 (回归任务，无激活函数)
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. 训练与评估函数 ---
def train_and_eval():
    # 超参数设置
    batch_size = 16
    num_epochs = 200
    lr = 0.01

    # 加载数据
    train_iter, test_iter, scaler_y = load_data(batch_size)
    input_size = 13  # 波士顿数据集有13个特征

    # 初始化网络、损失函数和优化器
    net = CustomNet(input_size)
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(net.parameters(), lr=lr)  # 使用 Adam 优化器

    # 记录训练过程
    train_losses = []
    test_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for X, y in train_iter:
            # 前向传播
            outputs = net(X).squeeze()  # 去掉多余的维度
            loss = criterion(outputs, y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 记录训练损失
        avg_train_loss = running_loss / len(train_iter)
        train_losses.append(avg_train_loss)

        # 验证阶段
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_iter:
                outputs = net(X).squeeze()
                loss = criterion(outputs, y)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_iter)
        test_losses.append(avg_test_loss)

        # 每 20 个 epoch 打印一次
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

    # --- 4. 结果可视化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 5. 精度评价指标 ---
    # 计算测试集上的 R^2 分数 (决定系数) 和 RMSE
    from sklearn.metrics import r2_score, mean_squared_error
    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            outputs = net(X).squeeze()
            y_true.extend(y.numpy())
            y_pred.extend(outputs.numpy())

    # 反标准化 (还原到原始房价尺度)
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_original = scaler_y.inverse_transform(y_true)
    y_pred_original = scaler_y.inverse_transform(y_pred)

    r2 = r2_score(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))

    print("\n--- 最终模型性能指标 ---")
    print(f"决定系数 R^2: {r2:.4f}")  # 越接近 1 越好
    print(f"均方根误差 RMSE: {rmse:.4f}")  # 越小越好
    print(f"平均房价预测误差: ±{rmse:.2f} (单位: 千美元)")

if __name__ == "__main__":
    # 检查 PyTorch 是否使用 CPU (你的 MacBook 是 Intel 芯片，所以应该是 CPU)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"设备: {'CPU'}")

    # 运行主程序
    train_and_eval()