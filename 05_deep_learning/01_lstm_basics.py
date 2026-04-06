# ============================================================
# 第一课：LSTM 入门 — 用正弦波理解 LSTM 的基本流程
# 学习目标：
#   1. 理解 LSTM 的输入格式（三维数组）
#   2. 掌握"滑动窗口"构造时序数据的方法
#   3. 跑通 LSTM 从构建到预测的完整流程
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 55)
print("LSTM 入门 — 正弦波预测 (PyTorch版)")
print("=" * 55)


# ─────────────────────────────────────────────
# 【0. LSTM 是什么？（先有个概念）】
# ─────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════╗
║                   LSTM 是什么？                        ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  LSTM = Long Short-Term Memory（长短期记忆网络）         ║
║                                                      ║
║  它是 RNN（循环神经网络）的升级版                          ║
║  普通RNN的问题: 记不住太久之前的信息                        ║
║  LSTM的解决:  加了"记忆门"控制该记什么、忘什么               ║
║                                                      ║
║  打个比方:                                             ║
║    普通RNN = 读完就忘的学渣                              ║
║    LSTM    = 会做笔记的学霸                             ║
║              知道哪些重点要记，哪些旧笔记该扔               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
# 【1. 生成数据 — 正弦波】
# ─────────────────────────────────────────────
print("【1. 生成正弦波数据】")
print("-" * 40)

# 生成 500 个正弦波数据点
t = np.linspace(0, 50, 500)
data = np.sin(t)

print(f"数据量: {len(data)} 个点")
print(f"前10个值: {np.round(data[:10], 3)}")

# 画出来看看
plt.figure(figsize=(12, 4))
plt.plot(t, data, color='steelblue', linewidth=1)
plt.title('正弦波数据（要预测的曲线）')
plt.xlabel('时间')
plt.ylabel('值')
plt.tight_layout()
plt.savefig('D:/python-lstm-learning/05_deep_learning/01_sine_data.png', dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# 【2. 滑动窗口 — LSTM 最核心的概念】
# ─────────────────────────────────────────────
print("\n\n【2. 滑动窗口 — 把时序数据变成训练样本】")
print("-" * 40)

print("""
LSTM 不能直接吃一整条时间序列，它需要"一段一段"地看：

  原始数据: [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, ...]

  用前5个预测第6个（time_steps=5）:

  样本1: 输入 [0.1, 0.3, 0.5, 0.7, 0.9] → 预测 0.8
  样本2: 输入 [0.3, 0.5, 0.7, 0.9, 0.8] → 预测 0.6
  样本3: 输入 [0.5, 0.7, 0.9, 0.8, 0.6] → 预测 0.4
  样本4: 输入 [0.7, 0.9, 0.8, 0.6, 0.4] → 预测 0.2
  ...窗口一格一格往右滑，每次取5个预测下一个

  这就是"滑动窗口"！
""")

TIME_STEPS = 10  # 用前10个点预测第11个

def create_dataset(data, time_steps):
    """
    滑动窗口：把一维时序数据切成 LSTM 需要的格式

    输入: data = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, -0.1, ...]
    输入: time_steps = 5

    输出:
      X = [[0.1, 0.3, 0.5, 0.7, 0.9],     ← 第1个样本
           [0.3, 0.5, 0.7, 0.9, 0.8],     ← 第2个样本
           [0.5, 0.7, 0.9, 0.8, 0.6], ...]
      y = [0.8, 0.6, 0.4, ...]             ← 每个样本对应的预测目标
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


X, y = create_dataset(data, TIME_STEPS)

print(f"滑动窗口大小: {TIME_STEPS}")
print(f"生成样本数: {len(X)}")
print(f"\n第1个样本:")
print(f"  输入 X[0] = {np.round(X[0], 3)}")
print(f"  目标 y[0] = {y[0]:.3f}")
print(f"第2个样本:")
print(f"  输入 X[1] = {np.round(X[1], 3)}")
print(f"  目标 y[1] = {y[1]:.3f}")


# ─────────────────────────────────────────────
# 【3. LSTM 的输入格式 — 三维数组】
# ─────────────────────────────────────────────
print("\n\n【3. LSTM 的输入格式 — 为什么是三维？】")
print("-" * 40)

print("""
sklearn 的输入是二维: (样本数, 特征数)
LSTM 的输入必须是三维: (样本数, 时间步, 特征数)

  X_train.shape = (样本数, TIME_STEPS, 特征数)

  正弦波只有1个特征（值本身），所以特征数=1:
    X_train.shape = (392, 10, 1)
                    样本   时间步 特征

  如果是滑坡数据（降雨+水位+位移），特征数=3:
    X_train.shape = (62, 10, 3)
                    样本  时间步 3个特征

  这就是为什么需要 reshape !
""")

# 把 X 从 (样本, 时间步) 变成 (样本, 时间步, 1)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"reshape 前: {X.shape}")
print(f"reshape 后: ({X.shape[0]}, {X.shape[1]}, 1)")

# 按时间顺序划分（时序数据不能随机打乱！）
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n训练集: {X_train.shape[0]} 个样本（前80%时间）")
print(f"测试集: {X_test.shape[0]} 个样本（后20%时间）")
print(f"X_train 形状: {X_train.shape}  (样本, 时间步, 特征)")
print(f"y_train 形状: {y_train.shape}  (样本,)")


# ─────────────────────────────────────────────
# 【4. 构建 LSTM 模型 (PyTorch版)】
# ─────────────────────────────────────────────
print("\n\n【4. 构建 LSTM 模型 (PyTorch版)】")
print("-" * 40)

print("""
PyTorch 构建模型的方式：
  1. 定义一个类，继承 nn.Module
  2. 在 __init__ 里定义层
  3. 在 forward 里定义数据流动

模型结构：
  输入层 → LSTM层 → 全连接层 → 输出

  nn.LSTM(1, 50)    → 输入1个特征，50个隐藏单元（记忆单元）
  nn.Linear(50, 1)  → 从50维映射到1维输出
""")


class LSTMModel(nn.Module):
    """LSTM 模型定义"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层：把 LSTM 输出映射到预测值
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x 形状: (batch_size, time_steps, input_size)
        
        # LSTM 前向传播
        # out: (batch_size, time_steps, hidden_size)
        # hidden, cell: 隐藏状态和细胞状态（这里不需要）
        out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # 全连接层输出
        out = self.fc(out)  # (batch_size, 1)
        return out


# 创建模型
model = LSTMModel(input_size=1, hidden_size=50)
print("\n模型结构:")
print(model)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params}")


# ─────────────────────────────────────────────
# 【5. 训练准备】
# ─────────────────────────────────────────────
print("\n\n【5. 训练准备】")
print("-" * 40)

print("""
PyTorch 训练需要准备：
  1. 损失函数 (criterion)  → 怎么算误差
  2. 优化器 (optimizer)    → 怎么更新参数
  3. 数据加载器 (DataLoader) → 批量喂数据
""")

# 超参数
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 损失函数：均方误差
criterion = nn.MSELoss()

# 优化器：Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练轮数: {EPOCHS}")
print(f"批量大小: {BATCH_SIZE}")
print(f"学习率: {LEARNING_RATE}")
print(f"损失函数: MSE (均方误差)")
print(f"优化器: Adam")


# ─────────────────────────────────────────────
# 【6. 训练模型】
# ─────────────────────────────────────────────
print("\n\n【6. 训练模型】")
print("-" * 40)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()  # 训练模式
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        epoch_loss += loss.item()
    
    # 计算平均训练损失
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证（用测试集）
    model.eval()  # 评估模式
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss:.6f}")

print(f"\n最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")


# ─────────────────────────────────────────────
# 【7. 预测 & 评估】
# ─────────────────────────────────────────────
print("\n\n【7. 预测 & 评估】")
print("-" * 40)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy().flatten()

mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"MSE:  {mse:.6f}   ← 均方误差")
print(f"RMSE: {rmse:.6f}   ← 均方根误差")
print(f"R2:   {r2:.4f}     ← 拟合优度")

if r2 > 0.95:
    print("\n预测效果非常好！LSTM 学到了正弦波的规律")
elif r2 > 0.8:
    print("\n预测效果不错，再多训练几轮会更好")
else:
    print("\n效果一般，可以尝试增加 epochs 或调整模型结构")


# ─────────────────────────────────────────────
# 【8. 画图】
# ─────────────────────────────────────────────
print("\n\n【8. 画图】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LSTM 正弦波预测结果 (PyTorch)', fontsize=16, fontweight='bold')

# ① 左上：训练损失曲线
axes[0, 0].plot(train_losses, label='训练损失', color='steelblue')
axes[0, 0].plot(val_losses, label='验证损失', color='red')
axes[0, 0].set_xlabel('Epoch（训练轮数）')
axes[0, 0].set_ylabel('损失 (MSE)')
axes[0, 0].set_title('训练损失曲线')
axes[0, 0].legend()

# ② 右上：预测 vs 真实（折线对比）
axes[0, 1].plot(y_test, label='真实值', linewidth=1.5, color='black')
axes[0, 1].plot(y_pred, label='LSTM预测', linewidth=1.5,
                color='steelblue', linestyle='--')
axes[0, 1].set_xlabel('测试样本')
axes[0, 1].set_ylabel('值')
axes[0, 1].set_title(f'预测 vs 真实 (R2={r2:.4f})')
axes[0, 1].legend()

# ③ 左下：散点图（预测 vs 真实）
axes[1, 0].scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=30)
axes[1, 0].plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='完美预测线')
axes[1, 0].set_xlabel('真实值')
axes[1, 0].set_ylabel('预测值')
axes[1, 0].set_title('散点图（越接近红线越好）')
axes[1, 0].set_xlim(-1.2, 1.2)
axes[1, 0].set_ylim(-1.2, 1.2)
axes[1, 0].legend()

# ④ 右下：滑动窗口可视化
show_window = 15  # 只展示前15个数据点
for i in range(min(show_window, len(data) - TIME_STEPS)):
    if i < show_window - 1:
        axes[1, 1].plot(range(i, i + TIME_STEPS), data[i:i + TIME_STEPS],
                        color='lightblue', alpha=0.3, linewidth=2)
    else:
        # 最后一个窗口高亮
        axes[1, 1].plot(range(i, i + TIME_STEPS), data[i:i + TIME_STEPS],
                        color='steelblue', alpha=0.8, linewidth=3)
        axes[1, 1].scatter(i + TIME_STEPS, data[i + TIME_STEPS],
                           color='red', s=100, zorder=5, label='预测目标')
axes[1, 1].plot(data[:show_window + TIME_STEPS + 1], 'k-', linewidth=1.5, alpha=0.5)
axes[1, 1].set_title(f'滑动窗口示意 (window={TIME_STEPS})')
axes[1, 1].set_xlabel('时间')
axes[1, 1].set_ylabel('值')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/05_deep_learning/01_lstm_sine_result.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("图片已保存: 01_lstm_sine_result.png")


# ─────────────────────────────────────────────
# 【9. 小结】
# ─────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("本课小结")
print("=" * 55)
print("""
1. 滑动窗口（最核心！）
   把时间序列切成一段一段，每段预测下一个值
   X = [过去10步的数据]  → 预测 y = [第11步的值]

2. LSTM 输入格式（三维）
   (样本数, 时间步, 特征数)
   正弦波: (392, 10, 1)    ← 1个特征（值本身）
   滑坡:   (62, 10, 3)     ← 3个特征（降雨、水位、位移）

3. PyTorch LSTM 模型结构
   class LSTMModel(nn.Module):
       def __init__(self):
           self.lstm = nn.LSTM(input_size, hidden_size)
           self.fc = nn.Linear(hidden_size, output_size)
   
   比 Keras 稍微多写几行，但更灵活

4. 训练流程（PyTorch版）
   ① 定义损失函数: criterion = nn.MSELoss()
   ② 定义优化器: optimizer = torch.optim.Adam(model.parameters())
   ③ 训练循环:
      for epoch in range(EPOCHS):
          outputs = model(X)      # 前向传播
          loss = criterion(outputs, y)
          optimizer.zero_grad()   # 清空梯度
          loss.backward()         # 反向传播
          optimizer.step()        # 更新参数

5. PyTorch vs TensorFlow/Keras
   PyTorch: 更灵活，学术界主流，支持 Python 3.13
   Keras:   更封装，初学者友好，但 Python 3.13 暂不支持
""")
