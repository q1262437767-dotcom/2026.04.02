"""
第二课：多图布局与毕设实战图
============================
上一课学了基础图表，这课直接练毕设答辩要用的图：

1. 预测值 vs 真实值对比图（LSTM 最经典的图）
2. 训练损失曲线（训练过程诊断）
3. 特征相关性热力图（降雨/水位/位移之间的关系）
4. 组合大图：一键生成答辩用图
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 通用设置：中文显示
# ─────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文黑体
plt.rcParams['axes.unicode_minus'] = False         # 正常显示负号

# 固定随机种子，保证每次运行结果一致
np.random.seed(42)

# 生成模拟数据（72个月，2018-2023）
months = np.arange(72)

# 模拟真实位移（累积增长 + 季节波动）
true_displacement = np.cumsum(np.random.uniform(0.5, 3.0, 72))
true_displacement += np.sin(months * np.pi / 6) * 2  # 加点季节波动

# 模拟 LSTM 预测值（跟真实值差不多，但有误差）
prediction_error = np.random.normal(0, 1.5, 72)
predicted = true_displacement + prediction_error

# 模拟训练损失曲线（越训练越小，最后趋于平稳）
train_loss = 0.85 * np.exp(-np.arange(100) / 15) + 0.02 + np.random.normal(0, 0.01, 100)
val_loss = 0.85 * np.exp(-np.arange(100) / 12) + 0.05 + np.random.normal(0, 0.015, 100)

# 模拟特征数据
rainfall = np.random.uniform(20, 200, 72)
water_level = 155 + 20 * np.sin(months * np.pi / 6) + np.random.normal(0, 2, 72)
feature_data = np.column_stack([rainfall, water_level, true_displacement])


# ═══════════════════════════════════════════════
# 【1. 预测值 vs 真实值对比图】
#    ← LSTM 论文/毕设 必备图
# ═══════════════════════════════════════════════
print("【1. 预测值 vs 真实值对比图】")

fig, ax = plt.subplots(figsize=(12, 5))

# 画两条线
ax.plot(months, true_displacement, 'b-', linewidth=2, label='真实值', alpha=0.9)
ax.plot(months, predicted, 'r--', linewidth=1.5, label='LSTM预测值', alpha=0.8)

# 填充误差区域（真实值和预测值之间）
ax.fill_between(months, true_displacement, predicted,
                alpha=0.15, color='orange', label='预测误差')

# 加一条"训练集/测试集"的分界线
split_idx = 54  # 前54个月训练，后18个月测试
ax.axvline(x=split_idx, color='gray', linestyle=':', linewidth=1.5)
ax.text(split_idx + 1, max(true_displacement) * 0.95,
        '← 测试集 | 训练集 →', fontsize=9, color='gray')

# 计算精度指标（测试集部分）
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_true = true_displacement[split_idx:]
y_pred = predicted[split_idx:]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# 把精度指标写在图的右上角
metrics_text = f'测试集精度:\nRMSE = {rmse:.2f} mm\nMAE  = {mae:.2f} mm\nR²   = {r2:.4f}'
ax.text(0.02, 0.95, metrics_text,
        transform=ax.transAxes,          # 用图的相对坐标（0~1）
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('月份（2018年1月起）', fontsize=12)
ax.set_ylabel('累积位移 (mm)', fontsize=12)
ax.set_title('白水河滑坡 LSTM 位移预测结果', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/05_prediction_comparison.png', dpi=150)
plt.show()
print("图已保存: 05_prediction_comparison.png")
print(f"  RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")


# ═══════════════════════════════════════════════
# 【2. 训练损失曲线】
#    ← 判断模型有没有"学好"
# ═══════════════════════════════════════════════
print("\n【2. 训练损失曲线】")

fig, ax = plt.subplots(figsize=(10, 5))

epochs = np.arange(1, 101)
ax.plot(epochs, train_loss, 'b-', linewidth=1.5, label='训练损失')
ax.plot(epochs, val_loss, 'r-', linewidth=1.5, label='验证损失')

# 标出最佳 epoch（验证损失最低的那个点）
best_epoch = np.argmin(val_loss) + 1
best_val = val_loss[best_epoch - 1]
ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
ax.annotate(f'最佳: 第{best_epoch}轮\nloss={best_val:.4f}',
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 15, best_val + 0.15),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=9, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('训练轮数 (Epoch)', fontsize=12)
ax.set_ylabel('损失值 (Loss)', fontsize=12)
ax.set_title('模型训练过程', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/06_loss_curve.png', dpi=150)
plt.show()
print("图已保存: 06_loss_curve.png")


# ═══════════════════════════════════════════════
# 【3. 特征相关性热力图】
#    ← 看降雨、水位、位移之间有没有关系
# ═══════════════════════════════════════════════
print("\n【3. 特征相关性热力图】")

# 计算相关系数矩阵
corr_matrix = np.corrcoef(feature_data, rowvar=False)
feature_names = ['降雨量', '水位', '位移']

fig, ax = plt.subplots(figsize=(7, 6))

# 画热力图
im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)

# 在每个格子里写上数字
for i in range(3):
    for j in range(3):
        val = corr_matrix[i, j]
        # 根据背景色选择文字颜色（深色背景用白色字）
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=14, fontweight='bold', color=text_color)

# 坐标轴标签
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(feature_names, fontsize=12)
ax.set_yticklabels(feature_names, fontsize=12)

# 加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('相关系数', fontsize=11)

ax.set_title('特征相关性分析', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('D:/python-lstm-learning/03_visualization/07_correlation_heatmap.png', dpi=150)
plt.show()
print("图已保存: 07_correlation_heatmap.png")


# ═══════════════════════════════════════════════
# 【4. 组合大图：答辩一键生成】
#    ← 把上面三张图 + 误差分布 拼在一起
# ═══════════════════════════════════════════════
print("\n【4. 答辩组合大图】")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('白水河滑坡 LSTM 位移预测 — 综合分析', fontsize=18, fontweight='bold', y=0.98)

# 用 gridspec 自定义布局（比 subplots 更灵活）
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ── 左上：预测对比 ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(months, true_displacement, 'b-', linewidth=1.5, label='真实值')
ax1.plot(months, predicted, 'r--', linewidth=1, label='LSTM预测')
ax1.axvline(x=split_idx, color='gray', linestyle=':', linewidth=1)
ax1.fill_between(months, true_displacement, predicted, alpha=0.1, color='orange')
ax1.text(0.02, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f} mm',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.set_title('预测值 vs 真实值', fontsize=13)
ax1.set_xlabel('月份')
ax1.set_ylabel('位移 (mm)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── 右上：损失曲线 ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_loss, 'b-', linewidth=1.2, label='训练损失')
ax2.plot(epochs, val_loss, 'r-', linewidth=1.2, label='验证损失')
ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6)
ax2.set_title(f'训练过程（最佳: 第{best_epoch}轮）', fontsize=13)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── 左下：热力图 ──
ax3 = fig.add_subplot(gs[1, 0])
im3 = ax3.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        val = corr_matrix[i, j]
        tc = 'white' if abs(val) > 0.5 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=13, fontweight='bold', color=tc)
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(feature_names)
ax3.set_yticklabels(feature_names)
plt.colorbar(im3, ax=ax3, shrink=0.8)
ax3.set_title('特征相关性', fontsize=13)

# ── 右下：误差分布直方图 ──
ax4 = fig.add_subplot(gs[1, 1])
errors = y_true - y_pred
ax4.hist(errors, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='零误差线')
ax4.set_title('测试集预测误差分布', fontsize=13)
ax4.set_xlabel('误差 (mm)')
ax4.set_ylabel('频次')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.savefig('D:/python-lstm-learning/03_visualization/08_comprehensive_analysis.png', dpi=150)
plt.show()
print("图已保存: 08_comprehensive_analysis.png")


# ═══════════════════════════════════════════════
# 【知识点总结】
# ═══════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════╗
║              第二课 知识点总结                      ║
╠══════════════════════════════════════════════════╣
║                                                   ║
║  📊 fill_between()  → 填充两条线之间的区域          ║
║     常用来可视化误差范围                            ║
║                                                   ║
║  📈 axvline()       → 画一条竖线                   ║
║     用来标记训练集/测试集分界                       ║
║                                                   ║
║  📝 transform=ax.transAxes → 用相对坐标放文字       ║
║     (0,0)=左下角, (1,1)=右上角                    ║
║                                                   ║
║  🔥 imshow()        → 画热力图（矩阵可视化）        ║
║     常用于相关性矩阵、混淆矩阵                      ║
║                                                   ║
║  📐 gridspec()      → 自定义多图布局               ║
║     比 subplots 更灵活，可以大小不一                ║
║                                                   ║
║  📉 hist()          → 直方图，看数据分布            ║
║     常看误差分布是否正态                            ║
║                                                   ║
║  📏 精度指标:                                      ║
║     RMSE → 预测误差大小（越小越好）                 ║
║     MAE  → 平均绝对误差（越小越好）                 ║
║     R²   → 拟合优度（越接近1越好）                  ║
║                                                   ║
╚══════════════════════════════════════════════════╝
""")
