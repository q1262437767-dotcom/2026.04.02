"""
第三课：滑坡可视化综合实战
==========================
用 02_data_processing/landslide_data.csv 的真实数据，
画出毕设答辩用的完整图集。

这节课 = 把前两课学的画图技能，用在真实数据上。
最终产出：一套可以直接放论文/PPT 的图。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# 0. 通用设置
# ─────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读入真实数据
df = pd.read_csv('D:/python-lstm-learning/02_data_processing/landslide_data.csv')
df['date'] = pd.to_datetime(df['date'])

print("数据概况:")
print(f"  时间范围: {df['date'].iloc[0].strftime('%Y-%m')} ~ {df['date'].iloc[-1].strftime('%Y-%m')}")
print(f"  共 {len(df)} 个月的数据")
print(f"  列: {list(df.columns)}\n")


# ═══════════════════════════════════════════════
# 【1. 三特征时间序列对比图】
#    ← 论文必备：一图看清三个变量的变化趋势
# ═══════════════════════════════════════════════
print("【1. 三特征时间序列对比图】")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('白水河滑坡监测数据时间序列', fontsize=16, fontweight='bold')

# 三个子图
features = [
    ('rainfall', '降雨量 (mm)', 'royalblue', True),
    ('water_level', '库水位 (m)', 'forestgreen', False),
    ('displacement', '累积位移 (mm)', 'crimson', False),
]

for ax, (col, ylabel, color, bar) in zip(axes, features):
    if bar:
        # 降雨量用柱状图更直观
        ax.bar(df['date'], df[col], width=20, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    else:
        ax.plot(df['date'], df[col], color=color, linewidth=2)
        ax.fill_between(df['date'], df[col], alpha=0.1, color=color)
    
    # 标出最大值和最小值
    max_idx = df[col].idxmax()
    min_idx = df[col].idxmin()
    ax.annotate(f'{df.loc[max_idx, col]:.1f}',
                xy=(df.loc[max_idx, 'date'], df.loc[max_idx, col]),
                fontsize=8, color=color, fontweight='bold',
                ha='center', va='bottom',
                xytext=(0, 8), textcoords='offset points')
    ax.annotate(f'{df.loc[min_idx, col]:.1f}',
                xy=(df.loc[min_idx, 'date'], df.loc[min_idx, col]),
                fontsize=8, color='gray',
                ha='center', va='top',
                xytext=(0, -12), textcoords='offset points')
    
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    # 在每个子图左边放标签
    ax.text(0.01, 0.92, ['(a)', '(b)', '(c)'][features.index((col, ylabel, color, bar))],
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[-1].set_xlabel('时间', fontsize=12)
plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/09_time_series.png', dpi=150)
plt.show()
print("图已保存: 09_time_series.png")


# ═══════════════════════════════════════════════
# 【2. 特征相关性热力图（真实数据版）】
# ═══════════════════════════════════════════════
print("\n【2. 特征相关性热力图】")

cols = ['rainfall', 'water_level', 'displacement']
corr = df[cols].corr()
labels = ['降雨量', '库水位', '位移']

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)

for i in range(3):
    for j in range(3):
        val = corr.values[i, j]
        tc = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=14, fontweight='bold', color=tc)

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('白水河滑坡监测数据相关性分析', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/10_correlation_real.png', dpi=150)
plt.show()

# 打印相关性结果（文字版）
print("相关系数矩阵:")
for i, r in enumerate(labels):
    print(f"  {r}: ", end="")
    for j, c in enumerate(labels):
        print(f"{corr.values[i,j]:.3f}  ", end="")
    print()

# 分析哪个特征跟位移最相关
dis_corr = corr['displacement'].drop('displacement')
strongest = dis_corr.abs().idxmax()
print(f"\n→ 与位移相关性最强的特征: {strongest} ({corr.loc[strongest, 'displacement']:.3f})")


# ═══════════════════════════════════════════════
# 【3. 位移速率分析图】
#    ← 位移增量 = 本月位移 - 上月位移（滑得多快）
# ═══════════════════════════════════════════════
print("\n【3. 位移速率分析图】")

df['位移增量'] = df['displacement'].diff()
df['位移速率'] = df['位移增量'] / df['date'].diff().dt.days  # mm/天

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('白水河滑坡位移速率分析', fontsize=16, fontweight='bold')

# 上图：位移增量柱状图（正=滑动，负=回弹）
colors = ['crimson' if v >= 0 else 'steelblue' for v in df['位移增量'].fillna(0)]
ax1.bar(df['date'], df['位移增量'], width=20, color=colors, alpha=0.7, edgecolor='white')
ax1.axhline(y=0, color='black', linewidth=0.8)
avg_rate = df['位移增量'].mean()
ax1.axhline(y=avg_rate, color='orange', linestyle='--', linewidth=1.5, label=f'月均增量: {avg_rate:.2f} mm')
ax1.set_ylabel('月位移增量 (mm)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 下图：累计位移（在原图上标注加速/减速段）
ax2.plot(df['date'], df['displacement'], color='crimson', linewidth=2)
ax2.fill_between(df['date'], df['displacement'], alpha=0.15, color='crimson')

# 标出位移增量最大的3个月
top3 = df.nlargest(3, '位移增量')
for _, row in top3.iterrows():
    ax2.annotate(f"{row['位移增量']:.1f}mm",
                 xy=(row['date'], row['displacement']),
                 xytext=(10, 15), textcoords='offset points',
                 fontsize=8, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=1),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

ax2.set_ylabel('累积位移 (mm)', fontsize=11)
ax2.set_xlabel('时间', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/11_displacement_rate.png', dpi=150)
plt.show()
print("图已保存: 11_displacement_rate.png")


# ═══════════════════════════════════════════════
# 【4. 三特征归一化对比图】
#    ← 三个变量量级不同，归一化后放一起比较趋势
# ═══════════════════════════════════════════════
print("\n【4. 三特征归一化对比图】")

scaler = MinMaxScaler()
data_norm = scaler.fit_transform(df[['rainfall', 'water_level', 'displacement']])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['date'], data_norm[:, 0], label='降雨量', color='royalblue', alpha=0.8)
ax.plot(df['date'], data_norm[:, 1], label='库水位', color='forestgreen', alpha=0.8)
ax.plot(df['date'], data_norm[:, 2], label='位移', color='crimson', linewidth=2, alpha=0.9)

ax.set_ylabel('归一化值 (0~1)', fontsize=12)
ax.set_xlabel('时间', fontsize=12)
ax.set_title('三特征归一化对比（观察趋势同步性）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/12_normalized_comparison.png', dpi=150)
plt.show()
print("图已保存: 12_normalized_comparison.png")

print("\n→ 归一化后可以看出三者的趋势是否有同步性")
print("  如果位移的峰值总是跟着降雨/水位的峰值出现 → 说明相关性强")


# ═══════════════════════════════════════════════
# 【5. 答辩综合大图（一页PPT）】
#    ← 把关键图拼成一张，答辩展示用
# ═══════════════════════════════════════════════
print("\n【5. 答辩综合大图】")

fig = plt.figure(figsize=(16, 14))
fig.suptitle('白水河滑坡多源监测数据综合分析', fontsize=18, fontweight='bold', y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                      height_ratios=[1, 1, 0.8])  # 上面两行高，下面一行矮

# ── (a) 三特征时间序列 ──
ax_a = fig.add_subplot(gs[0, :])  # 占满整行
ax_a.bar(df['date'], df['rainfall'], width=20, color='royalblue', alpha=0.5, label='降雨量 (mm)')
ax_a2 = ax_a.twinx()  # 双Y轴
ax_a2.plot(df['date'], df['water_level'], color='forestgreen', linewidth=1.5, label='库水位 (m)')
ax_a2.plot(df['date'], df['displacement'], color='crimson', linewidth=2, label='位移 (mm)')
ax_a.set_ylabel('降雨量 (mm)', color='royalblue', fontsize=11)
ax_a2.set_ylabel('水位(m) / 位移(mm)', fontsize=11)
ax_a.set_title('(a) 监测数据时间序列（双Y轴）', fontsize=12)
lines1, labels1 = ax_a.get_legend_handles_labels()
lines2, labels2 = ax_a2.get_legend_handles_labels()
ax_a.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
ax_a.grid(True, alpha=0.2)

# ── (b) 热力图 ──
ax_b = fig.add_subplot(gs[1, 0])
im_b = ax_b.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        val = corr.values[i, j]
        tc = 'white' if abs(val) > 0.5 else 'black'
        ax_b.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=12, fontweight='bold', color=tc)
ax_b.set_xticks([0, 1, 2]); ax_b.set_yticks([0, 1, 2])
ax_b.set_xticklabels(labels, fontsize=10); ax_b.set_yticklabels(labels, fontsize=10)
plt.colorbar(im_b, ax=ax_b, shrink=0.8)
ax_b.set_title('(b) 特征相关性分析', fontsize=12)

# ── (c) 位移增量 ──
ax_c = fig.add_subplot(gs[1, 1])
colors_c = ['crimson' if v >= 0 else 'steelblue' for v in df['位移增量'].fillna(0)]
ax_c.bar(df['date'], df['位移增量'], width=20, color=colors_c, alpha=0.7, edgecolor='white')
ax_c.axhline(y=0, color='black', linewidth=0.8)
ax_c.axhline(y=avg_rate, color='orange', linestyle='--', linewidth=1.2, label=f'均值: {avg_rate:.2f} mm')
ax_c.set_ylabel('月位移增量 (mm)', fontsize=11)
ax_c.set_title('(c) 位移增量分析', fontsize=12)
ax_c.legend(fontsize=9)
ax_c.grid(True, alpha=0.3)

# ── (d) 归一化对比 ──
ax_d = fig.add_subplot(gs[2, :])  # 占满整行
ax_d.plot(df['date'], data_norm[:, 0], label='降雨量', color='royalblue', alpha=0.7)
ax_d.plot(df['date'], data_norm[:, 1], label='库水位', color='forestgreen', alpha=0.7)
ax_d.plot(df['date'], data_norm[:, 2], label='位移', color='crimson', linewidth=2)
ax_d.set_ylabel('归一化值', fontsize=11)
ax_d.set_xlabel('时间', fontsize=11)
ax_d.set_title('(d) 三特征归一化趋势对比', fontsize=12)
ax_d.legend(fontsize=9, loc='upper left')
ax_d.grid(True, alpha=0.3)
ax_d.set_ylim(-0.05, 1.05)

plt.savefig('D:/python-lstm-learning/03_visualization/13_comprehensive_real.png', dpi=150)
plt.show()
print("图已保存: 13_comprehensive_real.png")


# ═══════════════════════════════════════════════
# 【知识点总结】
# ═══════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════╗
║           第三课 知识点总结                        ║
╠══════════════════════════════════════════════════╣
║                                                   ║
║  📊 sharex=True       → 多个子图共享X轴            ║
║     时间序列图必备，X轴对齐更整齐                  ║
║                                                   ║
║  📊 twinx()           → 双Y轴                     ║
║     左右各一个Y轴，显示不同量级的数据              ║
║                                                   ║
║  📊 diff()             → 算位移增量（Pandas）       ║
║  📊 nlargest(3, col)  → 取最大的3行               ║
║  📊 fill_between()     → 曲线下方填充颜色           ║
║                                                   ║
║  📊 gridspec + height_ratios → 子图大小可不一样     ║
║                                                   ║
║  🔑 答辩画图清单:                                  ║
║     ✅ 三特征时间序列                               ║
║     ✅ 相关性热力图                                 ║
║     ✅ 位移速率/增量分析                             ║
║     ✅ 归一化趋势对比                               ║
║     → 第5课LSTM训练后加上预测对比图就齐了           ║
║                                                   ║
╚══════════════════════════════════════════════════╝
""")
