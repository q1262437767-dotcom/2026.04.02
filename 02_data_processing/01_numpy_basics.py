# ============================================================
# NumPy 基础 - 数据处理核心工具
# 学习目标：掌握数组创建、运算、索引、统计
# ============================================================

import numpy as np

print("=" * 50)
print("【1. 创建数组】")
print("=" * 50)

# 从列表创建
a = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {a}")
print(f"类型: {type(a)}")
print(f"数据类型: {a.dtype}")
print(f"形状: {a.shape}")

# 常用创建方式
print("\n常用创建方式:")
print(f"zeros:  {np.zeros(5)}")             # 全0
print(f"ones:   {np.ones(5)}")              # 全1
print(f"range:  {np.arange(0, 10, 2)}")    # 0到10步长2
print(f"linspace: {np.linspace(0, 1, 5)}") # 0到1均匀分5个

# 二维数组（矩阵）
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"\n二维数组:\n{matrix}")
print(f"形状: {matrix.shape}")   # (行, 列)


print("\n" + "=" * 50)
print("【2. 数组运算】")
print("=" * 50)

a = np.array([10, 20, 30, 40, 50])
b = np.array([1, 2, 3, 4, 5])

print(f"a + b  = {a + b}")      # 对应元素相加
print(f"a * b  = {a * b}")      # 对应元素相乘
print(f"a / 10 = {a / 10}")     # 每个元素除以10
print(f"a ** 2 = {a ** 2}")     # 每个元素平方
print(f"a > 25: {a > 25}")      # 每个元素比较，返回布尔数组


print("\n" + "=" * 50)
print("【3. 索引与切片】")
print("=" * 50)

data = np.array([5.2, 8.1, 12.3, 6.7, 18.5, 3.4, 15.2])

print(f"原数组: {data}")
print(f"第0个:  {data[0]}")
print(f"最后一个: {data[-1]}")
print(f"前3个:  {data[:3]}")
print(f"第2到4个: {data[2:5]}")

# 布尔索引（重要！）
mask = data > 10
print(f"\n大于10的mask: {mask}")
print(f"大于10的值:   {data[mask]}")
# 简写
print(f"简写方式:     {data[data > 10]}")


print("\n" + "=" * 50)
print("【4. 统计函数】")
print("=" * 50)

displacement = np.array([5.2, 8.1, 12.3, 6.7, 18.5, 3.4, 15.2, 9.8, 11.1, 7.6])

print(f"数据: {displacement}")
print(f"最大值: {np.max(displacement):.2f}")
print(f"最小值: {np.min(displacement):.2f}")
print(f"平均值: {np.mean(displacement):.2f}")
print(f"中位数: {np.median(displacement):.2f}")
print(f"标准差: {np.std(displacement):.2f}")
print(f"总和:   {np.sum(displacement):.2f}")

# 找最大值在哪个位置
max_idx = np.argmax(displacement)
print(f"\n最大值 {displacement[max_idx]} 在第 {max_idx} 个位置")


print("\n" + "=" * 50)
print("【5. 实战：滑坡数据处理】")
print("=" * 50)

# 模拟6年月位移数据（72个月）
np.random.seed(42)
monthly_disp = np.cumsum(np.random.uniform(0.5, 3.0, 72))  # 累计位移
rainfall = np.random.uniform(20, 150, 72)                    # 月降雨量
water_level = np.random.uniform(155, 178, 72)                # 月均水位

print(f"数据长度: {len(monthly_disp)} 个月")
print(f"位移范围: {monthly_disp.min():.1f} ~ {monthly_disp.max():.1f} mm")
print(f"位移均值: {monthly_disp.mean():.1f} mm")

# 找高降雨月份（>100mm）的位移变化
high_rain_mask = rainfall > 100
high_rain_disp = monthly_disp[high_rain_mask]
print(f"\n高降雨月份数量: {high_rain_mask.sum()} 个月")
print(f"高降雨月平均位移: {high_rain_disp.mean():.1f} mm")
print(f"整体平均位移:     {monthly_disp.mean():.1f} mm")

# 归一化（0-1之间）
disp_min = monthly_disp.min()
disp_max = monthly_disp.max()
disp_normalized = (monthly_disp - disp_min) / (disp_max - disp_min)
print(f"\n归一化后范围: {disp_normalized.min():.2f} ~ {disp_normalized.max():.2f}")
print(f"归一化前5个月: {monthly_disp[:5].round(2)}")
print(f"归一化后前5个: {disp_normalized[:5].round(4)}")

# TODO: 计算每年（12个月）的平均位移，共6年
# 提示：用 reshape 把72个数变成 (6, 12) 的矩阵，再按行求平均
yearly = monthly_disp.reshape(6, 12)
yearly_avg = np.mean(yearly, axis=1)
for i, avg in enumerate(yearly_avg):
    print(f"\n第{i+1}年平均位移: {avg:.1f} mm")
