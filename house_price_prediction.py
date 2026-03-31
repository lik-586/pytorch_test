import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()

# 1. 数据加载与预处理
print("正在加载加州房价数据集...")
data = fetch_california_housing()
X, y = data.data, data.target

# 划分训练集和测试集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 注意：对于回归任务的目标值y，也进行标准化有助于模型训练
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

# 2. 定义神经网络模型 (尝试新的结构)
class DeepRegressionNet(nn.Module):
    def __init__(self, input_dim=8):
        super(DeepRegressionNet, self).__init__()
        self.model = nn.Sequential(
            # 第一层：引入 BatchNorm 和 Dropout
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout 30%

            # 第二层：更深的网络结构
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 输出层：回归任务，输出维度为1
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数和优化器
model = DeepRegressionNet(input_dim=X_train.shape[1])
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Adam + L2正则化

# 3. 训练模型
num_epochs = 500
train_losses = []
test_losses = []

print(f"开始训练，共 {num_epochs} 个 Epoch...")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 记录训练损失
    train_losses.append(loss.item())

    # 在测试集上评估 (每50个epoch打印一次)
    if (epoch+1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.item())

            # 反标准化预测结果以便计算指标
            pred_scaled = test_outputs.numpy()
            y_test_pred = scaler_y.inverse_transform(pred_scaled)
            y_test_true = scaler_y.inverse_transform(y_test_tensor.numpy())

            rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
            r2 = r2_score(y_test_true, y_test_pred)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, '
              f'Test Loss: {test_loss.item():.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}')

# 4. 最终评估与绘图
model.eval()
with torch.no_grad():
    final_outputs = model(X_test_tensor)
    final_pred_scaled = final_outputs.numpy()
    final_true = y_test_tensor.numpy()

    # 反标准化以计算最终指标
    y_pred = scaler_y.inverse_transform(final_pred_scaled)
    y_true = scaler_y.inverse_transform(final_true)

    final_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    final_r2 = r2_score(y_true, y_pred)

print("\n" + "="*30)
print("训练完成！")
print(f"最终测试集 RMSE: {final_rmse:.4f}")
print(f"最终测试集 R2 Score: {final_r2:.4f}")
print("="*30)

# 可视化预测结果 vs 真实值
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('California Housing Prediction (True vs Predicted)')
plt.legend()
plt.grid(True)
plt.show()