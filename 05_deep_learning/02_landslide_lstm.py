# ============================================================
# 第二课：LSTM 滑坡位移预测 — 多特征输入
# 学习目标：
#   1. 学会用多个特征（降雨+水位+位移）预测未来位移
#   2. 对比单特征 vs 多特征的预测效果
#   3. 掌握多特征滑动窗口的构造方法
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("LSTM 滑坡位移预测 — 多特征输入")
print("=" * 60)


# ─────────────────────────────────────────────
# 【1. 加载并查看数据】
# ─────────────────────────────────────────────
print("\n【1. 加载滑坡数据】")
print("-" * 40)

# 读取之前生成的模拟滑坡数据
df = pd.read_csv('D:/python-lstm-learning/02_data_processing/landslide_data.csv')

print(f"数据形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")
print(f"\n前5行:")
print(df.head())
print(f"\n数据统计:")
print(df.describe())


# ─────────────────────────────────────────────
# 【2. 数据归一化（LSTM 必须做！）】
# ─────────────────────────────────────────────
print("\n\n【2. 数据归一化】")
print("-" * 40)

print("""
为什么 LSTM 需要归一化？
  降雨量范围: 50~250 mm
  水位范围:   145~175 m
  位移范围:   0~50 mm
  
  三个特征的数值差距很大！
  不归一化 → LSTM 会被大数值的特征"带偏"
  归一化后 → 所有特征变成 0~1 之间，公平对待
""")

# 提取特征
features = ['rainfall', 'water_level', 'displacement']
data = df[features].values

# 对全部特征归一化（LSTM 输入用）
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(data)

# 单独对位移列保存 scaler（反归一化时还原预测值用）
scaler_disp = MinMaxScaler()
scaler_disp.fit(data[:, 2].reshape(-1, 1))  # 只 fit 位移列

print("归一化前（前3行）:")
print(data[:3])
print("\n归一化后（前3行）:")
print(X_scaled[:3])


# ─────────────────────────────────────────────
# 【3. 多特征滑动窗口】
# ─────────────────────────────────────────────
print("\n\n【3. 多特征滑动窗口 — 关键变化！】")
print("-" * 40)

print("""
上一课（正弦波）的滑动窗口:
  输入: [0.1, 0.3, 0.5, 0.7, 0.9]     ← 1个特征
  预测: 0.8

这一课（滑坡）的滑动窗口:
  输入: [降雨50, 水位152, 位移1.7,     ← 3个特征
         降雨70, 水位154, 位移2.3,
         ...
         降雨120, 水位158, 位移5.1]
  预测: 位移 6.2

  每个时间步不再是1个数，而是3个数（降雨+水位+位移）
  
  X 形状: (样本数, 时间步, 3)  ← 最后一维是特征数
""")

TIME_STEPS = 10

def create_multivariate_dataset(data, time_steps, target_col):
    """
    多特征滑动窗口
    
    参数:
        data: 归一化后的数据 (样本数, 特征数)
        time_steps: 时间步长
        target_col: 要预测的目标列索引（位移=2）
    
    返回:
        X: (样本数, 时间步, 特征数)
        y: (样本数,)
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])          # 前10步的全部特征
        y.append(data[i + time_steps, target_col])  # 第11步的位移
    return np.array(X), np.array(y)


# 目标列索引：displacement 是第3列（索引2）
X, y = create_multivariate_dataset(X_scaled, TIME_STEPS, target_col=2)

print(f"时间步长: {TIME_STEPS}")
print(f"样本数: {len(X)}")
print(f"X 形状: {X.shape}  (样本, 时间步, 特征)")
print(f"y 形状: {y.shape}  (样本,)")
print(f"\n第1个样本的输入（前3步）:")
for step in range(3):
    print(f"  时间步{step+1}: 降雨={X[0, step, 0]:.3f}, 水位={X[0, step, 1]:.3f}, 位移={X[0, step, 2]:.3f}")
print(f"  → 预测目标: 位移={y[0]:.3f}")


# ─────────────────────────────────────────────
# 【4. 划分训练集和测试集】
# ─────────────────────────────────────────────
print("\n\n【4. 划分训练集和测试集】")
print("-" * 40)

# 时序数据：按时间顺序切，不能打乱！
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"训练集: {X_train.shape[0]} 个样本（前80%时间）")
print(f"测试集: {X_test.shape[0]} 个样本（后20%时间）")

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)


# ─────────────────────────────────────────────
# 【5. 构建 LSTM 模型】
# ─────────────────────────────────────────────
print("\n\n【5. 构建 LSTM 模型】")
print("-" * 40)

print("""
模型结构:
  输入 (batch, 10, 3)   ← 3个特征（降雨+水位+位移）
      ↓
  LSTM(3, 64)            ← 输入3个特征，64个隐藏单元
      ↓
  取最后一步输出
      ↓
  Linear(64, 1)          ← 输出1个值（预测位移）
      ↓
  预测值
""")


class LandslideLSTM(nn.Module):
    """滑坡位移预测 LSTM 模型"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(LandslideLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层（2层，比上一课多了一层，更"深"）
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # 取最后一个时间步的输出
        out = self.fc(out)
        return out


model = LandslideLSTM(input_size=3, hidden_size=64, num_layers=2)
print(f"模型结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params}")


# ─────────────────────────────────────────────
# 【6. 训练模型】
# ─────────────────────────────────────────────
print("\n\n【6. 训练模型】")
print("-" * 40)

EPOCHS = 100
LEARNING_RATE = 0.005

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 记录损失
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor).item()
        val_losses.append(val_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss:.6f}")

print(f"\n最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")


# ─────────────────────────────────────────────
# 【7. 预测 & 反归一化】
# ─────────────────────────────────────────────
print("\n\n【7. 预测 & 评估】")
print("-" * 40)

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy().flatten()

# 反归一化：把 0~1 的预测值还原成真实的位移值（mm）
y_test_real = scaler_disp.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler_disp.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 评估指标
mse = np.mean((y_test_real - y_pred_real) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test_real - y_pred_real))
r2 = 1 - np.sum((y_test_real - y_pred_real) ** 2) / np.sum((y_test_real - np.mean(y_test_real)) ** 2)

print(f"MSE:  {mse:.4f}  mm^2")
print(f"RMSE: {rmse:.4f}  mm")
print(f"MAE:  {mae:.4f}  mm")
print(f"R2:   {r2:.4f}")
print(f"\n平均预测误差: {rmse:.2f} mm")


# ─────────────────────────────────────────────
# 【8. 画图】
# ─────────────────────────────────────────────
print("\n\n【8. 可视化结果】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LSTM 滑坡位移预测结果（多特征）', fontsize=16, fontweight='bold')

# ① 左上：训练损失曲线
axes[0, 0].plot(train_losses, label='训练损失', color='steelblue', linewidth=1)
axes[0, 0].plot(val_losses, label='验证损失', color='red', linewidth=1)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('损失 (MSE)')
axes[0, 0].set_title('训练损失曲线')
axes[0, 0].legend()

# ② 右上：预测 vs 真实（折线）
axes[0, 1].plot(y_test_real, label='真实位移', linewidth=1.5, color='black')
axes[0, 1].plot(y_pred_real, label='LSTM预测', linewidth=1.5, color='steelblue', linestyle='--')
axes[0, 1].set_xlabel('测试样本')
axes[0, 1].set_ylabel('位移 (mm)')
axes[0, 1].set_title(f'预测 vs 真实 (R2={r2:.4f}, RMSE={rmse:.2f}mm)')
axes[0, 1].legend()

# ③ 左下：散点图
axes[1, 0].scatter(y_test_real, y_pred_real, alpha=0.6, color='steelblue', s=40)
max_val = max(y_test_real.max(), y_pred_real.max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')
axes[1, 0].set_xlabel('真实位移 (mm)')
axes[1, 0].set_ylabel('预测位移 (mm)')
axes[1, 0].set_title('散点图（越接近红线越好）')
axes[1, 0].legend()

# ④ 右下：误差分布
errors = y_pred_real - y_test_real
axes[1, 1].hist(errors, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('预测误差 (mm)')
axes[1, 1].set_ylabel('频次')
axes[1, 1].set_title(f'误差分布 (均值={np.mean(errors):.2f}mm)')

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/05_deep_learning/02_landslide_lstm_result.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("图片已保存: 02_landslide_lstm_result.png")


# ─────────────────────────────────────────────
# 【9. 小结】
# ─────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("本课小结")
print("=" * 60)
print("""
1. 归一化（必须做！）
   LSTM 对输入数据的大小很敏感
   不同特征的数值范围差很多时，必须先归一化到 0~1
   预测完之后要用 inverse_transform 反归一化还原真实值

2. 多特征滑动窗口
   上一课: X 形状 = (样本, 时间步, 1)     ← 正弦波只有1个特征
   这节课: X 形状 = (样本, 时间步, 3)     ← 降雨+水位+位移
   改变的只是特征数，滑动窗口的逻辑完全一样

3. 模型变化
   上一课: LSTM(1, 50) + Linear(50, 1)    ← 1个特征输入
   这节课: LSTM(3, 64) + Linear(64, 1)    ← 3个特征输入
   还加了第2层 LSTM 和 dropout，模型更"深"

4. 反归一化
   LSTM 输出的是 0~1 之间的值
   要还原成真实的位移值(mm)，需要 scaler.inverse_transform()

5. 新增评估指标
   MAE（平均绝对误差）: |预测-真实| 的平均值，更直观
   "平均每次预测差了 X mm"，答辩时比 RMSE 更好解释
""")
