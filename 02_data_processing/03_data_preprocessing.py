# ============================================================
# 数据预处理实战 - 直接对应毕设流程
# 学习目标：走一遍真实的数据清洗 → 归一化 → 滑动窗口流程
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("=" * 50)
print("数据预处理实战 — 毕设流程全模拟")
print("=" * 50)


# ─────────────────────────────────────────────
# 【1. 生成模拟数据（模拟你的白水河滑坡数据）】
# ─────────────────────────────────────────────
print("\n【1. 模拟滑坡监测数据】")
print("-" * 40)

np.random.seed(42)  # 固定随机种子，每次运行结果一样

# 模拟 72 个月（6年）的监测数据
months = pd.date_range('2018-01', periods=72, freq='ME')

# 降雨量：夏季多，冬季少（加点季节规律）
rainfall = np.random.uniform(30, 80, 72) + 80 * np.sin(np.linspace(0, 4*np.pi, 72))**2

# 库水位：每年蓄水-泄洪循环
water_level = 155 + 20 * np.sin(np.linspace(np.pi, 5*np.pi, 72))

# 累积位移（缓慢增长 + 受降雨和水位影响）
base_disp = np.cumsum(np.random.uniform(0.5, 2.0, 72))
displacement = base_disp + 0.02 * rainfall + 0.1 * water_level

df = pd.DataFrame({
    '日期': months,
    '降雨量(mm)': rainfall.round(1),
    '水位(m)': water_level.round(2),
    '位移(mm)': displacement.round(1)
})

print(df.head(10))
print(f"\n共 {len(df)} 条数据")


# ─────────────────────────────────────────────
# 【2. 数据检查 — 第一步永远先看数据长啥样】
# ─────────────────────────────────────────────
print("\n\n【2. 数据检查】")
print("-" * 40)

# 2.1 基本信息
print("--- df.info() ---")
df.info()

# 2.2 统计摘要
print("\n--- df.describe() ---")
print(df.describe())

# 2.3 检查缺失值
print(f"\n每列缺失值数量:")
print(df.isnull().sum())
# isnull() 每个单元格是空就返回 True
# .sum() 统计 True 的个数 = 缺失值数量


# ─────────────────────────────────────────────
# 【3. 数据清洗 — 处理异常值和缺失值】
# ─────────────────────────────────────────────
print("\n\n【3. 数据清洗】")
print("-" * 40)

# 3.1 人为制造一些缺失值（模拟真实数据）
df.loc[5, '降雨量(mm)'] = np.nan
df.loc[18, '位移(mm)'] = np.nan
df.loc[40, '水位(m)'] = np.nan
print(f"制造缺失值后:\n{df.isnull().sum()}")

# 3.2 填充缺失值
df['降雨量(mm)'] = df['降雨量(mm)'].ffill()   # 降雨用前值填充
df['位移(mm)'] = df['位移(mm)'].ffill()        # 位移用前值填充（连续数据）
df['水位(m)'] = df['水位(m)'].ffill()          # 水位用前值填充
print(f"\n填充后:\n{df.isnull().sum()}")
print("✅ 所有缺失值已处理")

# 3.3 异常值检测（用 3σ 原则：超过均值±3倍标准差就是异常）
print("\n--- 异常值检测（3σ原则） ---")
for col in ['降雨量(mm)', '水位(m)', '位移(mm)']:
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
    if len(outliers) > 0:
        print(f"{col}: 发现 {len(outliers)} 个异常值")
    else:
        print(f"{col}: 无异常值 ✅")


# ─────────────────────────────────────────────
# 【4. 特征选择 — 只保留需要的列】
# ─────────────────────────────────────────────
print("\n\n【4. 特征选择】")
print("-" * 40)

# 提取数值列用于建模
feature_cols = ['降雨量(mm)', '水位(m)', '位移(mm)']
data = df[feature_cols].values  # 转成 NumPy 数组

print(f"特征列: {feature_cols}")
print(f"数据形状: {data.shape}  → {data.shape[0]}个样本 × {data.shape[1]}个特征")
print(f"前3行:\n{data[:3]}")


# ─────────────────────────────────────────────
# 【5. 归一化 — 把数据缩放到 0~1】
# ─────────────────────────────────────────────
print("\n\n【5. 归一化（MinMaxScaler）】")
print("-" * 40)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print(f"归一化前范围: [{data.min():.1f}, {data.max():.1f}]")
print(f"归一化后范围: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
print(f"\n归一化后前3行:\n{scaled_data[:3].round(4)}")

# 反归一化（预测完要还原成真实值）
original = scaler.inverse_transform(scaled_data[:3])
print(f"\n反归一化验证（应该和原始数据一致）:\n{original.round(1)}")


# ─────────────────────────────────────────────
# 【6. 滑动窗口 — 构造 LSTM 的输入输出】
# ─────────────────────────────────────────────
print("\n\n【6. 滑动窗口构造数据】")
print("-" * 40)

window_size = 3  # 用前3个月预测第4个月

def create_dataset(data, window_size):
    """
    用滑动窗口把一维时间序列变成 LSTM 需要的 X, y 格式
    data: 归一化后的数据 (样本数 × 特征数)
    window_size: 用几个时间步作为输入
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])      # 3个时间步的3个特征
        y.append(data[i + window_size])           # 第4个月的3个特征（主要预测位移）
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)

print(f"窗口大小: {window_size}")
print(f"X 形状: {X.shape}")
print(f"  → {X.shape[0]}个样本, {X.shape[1]}个时间步, {X.shape[2]}个特征")
print(f"y 形状: {y.shape}")

# 打印第1个样本，直观理解
print(f"\n第1个样本:")
print(f"  X[0] (前3个月数据):\n{X[0].round(4)}")
print(f"  y[0] (第4个月, 目标):\n{y[0].round(4)}")

print(f"\n第2个样本:")
print(f"  X[1] (往右滑一格):\n{X[1].round(4)}")
print(f"  y[1]:\n{y[1].round(4)}")


# ─────────────────────────────────────────────
# 【7. 划分训练集和测试集】
# ─────────────────────────────────────────────
print("\n\n【7. 划分训练集/测试集】")
print("-" * 40)

# 时序数据不能随机打乱！要按时间顺序切分
train_size = int(len(X) * 0.8)  # 前80%训练，后20%测试

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"训练集: {X_train.shape[0]} 个样本 (前80%)")
print(f"测试集: {X_test.shape[0]} 个样本 (后20%)")
print(f"\n⚠️ 注意: 时序数据按时间切分，不能随机打乱!")
print(f"   如果随机打乱，模型可能'看到未来' → 过拟合")


# ─────────────────────────────────────────────
# 【8. 总结 — 毕设数据处理完整流程】
# ─────────────────────────────────────────────
print("\n\n【8. 完整流程总结】")
print("-" * 40)
print("""
你毕设的数据处理流程就是这7步:

  1. 读取数据     → pd.read_excel()
  2. 检查数据     → df.info() + df.describe() + df.isnull().sum()
  3. 清洗数据     → ffill() 填缺失值 + 异常值处理
  4. 特征选择     → df[['降雨', '水位', '位移']].values
  5. 归一化       → MinMaxScaler().fit_transform()
  6. 滑动窗口     → create_dataset(data, window_size)
  7. 划分训练测试  → X[:split] / X[split:]

接下来把 X_train 喂给 LSTM 模型就能训练了！
""")


# ─────────────────────────────────────────────
# TODO 练习
# ─────────────────────────────────────────────

# TODO 1: 把 window_size 改成 6（用半年数据预测下一个月），看看 X 的形状怎么变
window_size = 6  # 把3改成6

# TODO 2: 试试只选两个特征 ['降雨量(mm)', '水位(m)']，看看数据处理流程有什么不同
# 原来：3个特征
feature_cols = ['降雨量(mm)', '水位(m)', '位移(mm)']

# 改成：2个特征
feature_cols = ['降雨量(mm)', '水位(m)']

# TODO 3（选做）: 在第3步清洗时，给某个异常月份的位移设为 99999，
#   然后用 df[col].clip(lower=...) 或手动替换来修复它
# 制造异常值
df.loc[30, '位移(mm)'] = 99999
print(f"异常值: 第30行位移 = {df.loc[30, '位移(mm)']}")

# 用 clip 限制范围（超出范围的值会被"剪"到边界）
lower = df['位移(mm)'].mean() - 3 * df['位移(mm)'].std()
upper = df['位移(mm)'].mean() + 3 * df['位移(mm)'].std()
df['位移(mm)'] = df['位移(mm)'].clip(lower=lower, upper=upper)
print(f"修复后: 第30行位移 = {df.loc[30, '位移(mm)']:.1f}")
