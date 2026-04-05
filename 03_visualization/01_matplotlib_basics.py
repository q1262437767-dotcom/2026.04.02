# ============================================================
# Matplotlib 基础 - 数据可视化
# 学习目标：掌握折线图、散点图、柱状图、多图布局
# ============================================================

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# 中文显示设置（否则中文会显示成方块）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

print("=" * 50)
print("Matplotlib 基础 - 数据可视化")
print("=" * 50)

# 模拟滑坡监测数据（跟上节课一样）
np.random.seed(42)
months = pd.date_range('2018-01', periods=36, freq='ME')
rainfall = (np.random.uniform(30, 80, 36) + 80 * np.sin(np.linspace(0, 2*np.pi, 36))**2).round(1)
water_level = (155 + 20 * np.sin(np.linspace(np.pi, 3*np.pi, 36))).round(2)
displacement = np.cumsum(np.random.uniform(0.5, 2.0, 36)).round(1)


# ─────────────────────────────────────────────
# 【1. 最基础的折线图】
# ─────────────────────────────────────────────
print("\n【1. 折线图 — 位移随时间变化】")

plt.figure(figsize=(10, 4))          # 创建画布，宽10英寸，高4英寸

plt.plot(months, displacement,
         color='steelblue',          # 线的颜色
         linewidth=2,                # 线宽
         marker='o',                 # 每个数据点画个圆圈
         markersize=4,               # 圆圈大小
         label='累积位移')            # 图例标签

plt.title('白水河滑坡累积位移（2018-2020）', fontsize=14)  # 标题
plt.xlabel('时间')                   # x轴标签
plt.ylabel('位移 (mm)')             # y轴标签
plt.legend()                         # 显示图例
plt.grid(True, alpha=0.3)            # 显示网格，透明度0.3
plt.tight_layout()                   # 自动调整布局，避免文字被截断
plt.savefig('01_displacement_trend.png', dpi=150)   # 保存图片
plt.show()
print("图已保存: 01_displacement_trend.png")


# ─────────────────────────────────────────────
# 【2. 散点图 — 看两个变量的关系】
# ─────────────────────────────────────────────
print("\n【2. 散点图 — 降雨量 vs 位移增量】")

# 计算位移增量
disp_diff = np.diff(displacement, prepend=displacement[0])

plt.figure(figsize=(7, 5))

plt.scatter(rainfall, disp_diff,
            c=water_level,           # 颜色按水位映射（越红水位越高）
            cmap='RdYlBu_r',         # 颜色方案：红-黄-蓝
            s=60,                    # 点的大小
            alpha=0.8,               # 透明度
            edgecolors='gray',       # 点的边框颜色
            linewidths=0.5)

plt.colorbar(label='库水位 (m)')     # 右边加个颜色条
plt.title('降雨量 vs 位移增量（颜色=水位）', fontsize=13)
plt.xlabel('降雨量 (mm)')
plt.ylabel('位移增量 (mm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_rainfall_vs_displacement.png', dpi=150)
plt.show()
print("图已保存: 02_rainfall_vs_displacement.png")


# ─────────────────────────────────────────────
# 【3. 柱状图 — 按季度统计降雨】
# ─────────────────────────────────────────────
print("\n【3. 柱状图 — 季度降雨量统计】")

# 按季度求和（4个季度×3年=12个季度）
quarters = ['Q1', 'Q2', 'Q3', 'Q4'] * 3
yearly = ['2018']*4 + ['2019']*4 + ['2020']*4
rain_reshaped = rainfall[:12*3].reshape(3, 4, 3).sum(axis=2)  # 3年×4季度，每季度3个月求和

x = np.arange(4)          # x轴位置 [0,1,2,3]
width = 0.25               # 每个柱子的宽度

plt.figure(figsize=(9, 5))
for i, year in enumerate(['2018', '2019', '2020']):
    plt.bar(x + i*width, rain_reshaped[i],
            width=width,
            label=year,
            alpha=0.85)

plt.xticks(x + width, ['Q1\n(春)', 'Q2\n(夏)', 'Q3\n(秋)', 'Q4\n(冬)'])
plt.title('各季度累计降雨量', fontsize=13)
plt.xlabel('季度')
plt.ylabel('降雨量 (mm)')
plt.legend(title='年份')
plt.grid(True, axis='y', alpha=0.3)  # 只显示y轴网格
plt.tight_layout()
plt.savefig('03_quarterly_rainfall.png', dpi=150)
plt.show()
print("图已保存: 03_quarterly_rainfall.png")


# ─────────────────────────────────────────────
# 【4. 多子图 — 一图看三个特征】
# ─────────────────────────────────────────────
print("\n【4. 多子图 — 三特征联合展示】")

fig, axes = plt.subplots(3, 1,           # 3行1列
                          figsize=(12, 8),
                          sharex=True)    # 共享x轴（对齐时间）

# 子图1：降雨量（柱状图）
axes[0].bar(months, rainfall, color='skyblue', alpha=0.8, label='降雨量')
axes[0].set_ylabel('降雨量 (mm)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# 子图2：库水位（折线图）
axes[1].plot(months, water_level, color='orange', linewidth=2, label='库水位')
axes[1].set_ylabel('水位 (m)')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# 子图3：累积位移（折线图）
axes[2].plot(months, displacement, color='crimson', linewidth=2, label='累积位移')
axes[2].fill_between(months, displacement, alpha=0.15, color='crimson')  # 填充阴影
axes[2].set_ylabel('位移 (mm)')
axes[2].set_xlabel('时间')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

fig.suptitle('白水河滑坡监测数据（2018-2020）', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('04_three_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("图已保存: 04_three_features.png")
print("\n✅ 这张图直接可以放进毕设！")


# ─────────────────────────────────────────────
# TODO 练习
# ─────────────────────────────────────────────

# TODO 1: 修改折线图的颜色（把 'steelblue' 改成 'green' 或 '#FF5733'）
#          看看支持哪几种颜色写法

# TODO 2: 给多子图里的位移曲线加上"最大值"标注
#          提示: max_idx = np.argmax(displacement)
#                axes[2].annotate(f'最大值\n{displacement[max_idx]:.1f}mm', ...)

# 找到最大值的位置
max_idx = np.argmax(displacement)
max_val = displacement[max_idx]
max_date = months[max_idx]

# 在位移子图上加标注
axes[2].annotate(
    f'最大值\n{max_val:.1f}mm',          # 标注文字
    xy=(max_date, max_val),              # 箭头指向的点（最大值位置）
    xytext=(max_date, max_val - 50),     # 文字显示的位置（往下偏移50）
    fontsize=9,
    color='red',
    arrowprops=dict(
        arrowstyle='->',                 # 箭头样式
        color='red',
        lw=1.5
    ),
    bbox=dict(
        boxstyle='round,pad=0.3',        # 文字加个圆角框
        facecolor='lightyellow',
        edgecolor='red',
        alpha=0.8
    )
)

# TODO 3（选做）: 把今天的四张图拼成一个 2×2 的大图保存
#          提示: plt.subplots(2, 2, figsize=(14, 10))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('白水河滑坡数据可视化总览', fontsize=16, fontweight='bold')

# ① 左上：折线图 - 位移趋势
axes[0, 0].plot(months, displacement, color='steelblue', linewidth=1.5)
axes[0, 0].set_title('累积位移趋势')
axes[0, 0].set_xlabel('时间')
axes[0, 0].set_ylabel('位移 (mm)')

# ② 右上：散点图 - 降雨 vs 位移
sc = axes[0, 1].scatter(rainfall, displacement, c=water_level, cmap='RdYlBu_r', alpha=0.7)
fig.colorbar(sc, ax=axes[0, 1], label='水位 (m)')
axes[0, 1].set_title('降雨 vs 位移（颜色=水位）')
axes[0, 1].set_xlabel('降雨量 (mm)')
axes[0, 1].set_ylabel('位移 (mm)')

# ③ 左下：柱状图 - 季度降雨（用整体降雨按季度聚合）
quarterly_rain = rainfall.reshape(-1, 3).mean(axis=1)
quarters = [f'Q{i+1}' for i in range(len(quarterly_rain))]
axes[1, 0].bar(quarters, quarterly_rain, color='skyblue', edgecolor='navy', alpha=0.8)
axes[1, 0].set_title('季度平均降雨量')
axes[1, 0].set_xlabel('季度')
axes[1, 0].set_ylabel('降雨量 (mm)')

# ④ 右下：多线图 - 三特征归一化对比
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(
    np.column_stack([rainfall, water_level, displacement])
)
axes[1, 1].plot(months, data_norm[:, 0], label='降雨量', alpha=0.8)
axes[1, 1].plot(months, data_norm[:, 1], label='水位', alpha=0.8)
axes[1, 1].plot(months, data_norm[:, 2], label='位移', alpha=0.8)
axes[1, 1].set_title('三特征归一化对比')
axes[1, 1].set_xlabel('时间')
axes[1, 1].set_ylabel('归一化值')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/overview_2x2.png', dpi=150, bbox_inches='tight')
plt.show()
