"""
05_deep_learning/03_lstm_optimization.py
========================================
第三课：LSTM 调参与模型优化

学习目标：
1. 了解过拟合 & 欠拟合，知道怎么判断
2. 掌握 Dropout —— 防止过拟合的利器
3. 掌握早停（Early Stopping）—— 自动找到最好的训练轮数
4. 掌握批量训练（Mini-batch）—— 让训练更稳
5. 学会对比实验 —— 调参的科学方法

前置：已学过 01_lstm_basics.py、02_landslide_lstm.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：准备数据（和第二课一样，只是提取为函数）
# ============================================================

def load_and_prepare_data(csv_path, time_steps=10):
    """
    加载滑坡数据，归一化，切成滑动窗口样本。
    
    返回值：
        X_train, y_train, X_test, y_test  — 归一化后的 tensor
        scaler_disp                        — 位移列的 scaler（用于反归一化）
        df                                 — 原始 DataFrame（画图用）
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    features = ['rainfall', 'water_level', 'displacement']
    data = df[features].values

    # 全局归一化（保证训练/测试用同一套参数）
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(data)

    # 单独保存位移列的 scaler，用于反归一化
    scaler_disp = MinMaxScaler()
    scaler_disp.fit(data[:, 2].reshape(-1, 1))

    # 构建滑动窗口样本
    X_list, y_list = [], []
    for i in range(len(X_scaled) - time_steps):
        X_list.append(X_scaled[i: i + time_steps])       # 前 time_steps 步 → 输入
        y_list.append(X_scaled[i + time_steps, 2])       # 第 time_steps+1 步的位移 → 标签

    X_arr = np.array(X_list, dtype=np.float32)   # (N, time_steps, 3)
    y_arr = np.array(y_list, dtype=np.float32)   # (N,)

    # 按 8:2 切训练/测试（时序数据不能随机打乱！）
    split = int(len(X_arr) * 0.8)
    X_train = torch.tensor(X_arr[:split])
    y_train = torch.tensor(y_arr[:split]).unsqueeze(1)  # (N, 1)
    X_test  = torch.tensor(X_arr[split:])
    y_test  = torch.tensor(y_arr[split:]).unsqueeze(1)

    return X_train, y_train, X_test, y_test, scaler_disp, df


# ============================================================
# 第二部分：定义两个模型 —— 普通版 vs 优化版
# ============================================================

# -----------------------------------------------
# 模型 A：基础版（和第二课一样，没有优化）
# -----------------------------------------------
class LSTMBasic(nn.Module):
    """
    两层 LSTM + 全连接，没有任何正则化。
    训练时间长了容易过拟合。
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # 输入形状：(batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)       # out: (batch, seq, hidden)
        out = out[:, -1, :]         # 取最后一步: (batch, hidden)
        return self.fc(out)         # (batch, 1)


# -----------------------------------------------
# 模型 B：优化版（加了 Dropout）
# -----------------------------------------------
class LSTMOptimized(nn.Module):
    """
    两层 LSTM + Dropout + 全连接。
    
    Dropout 的作用：
        训练时随机关掉 20% 的神经元，强迫模型不要"死记硬背"，
        学到更通用的规律，从而减少过拟合。
        
    注意：Dropout 只在训练时生效，预测时自动关掉。
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,        # ← 新增：层间 Dropout（num_layers>1 时才生效）
        )
        self.dropout = nn.Dropout(dropout)   # ← 新增：LSTM 输出后再 Dropout 一次
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)     # ← 新增：输出过 Dropout
        return self.fc(out)


# ============================================================
# 第三部分：训练函数（含早停）
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=200, lr=0.001, batch_size=16, patience=20):
    """
    训练模型，支持：
    - 批量训练（Mini-batch）：每次只用一小批数据更新，训练更稳定
    - 早停（Early Stopping）：验证集 loss 连续 patience 轮不降就停止
    
    参数：
        patience : int
            早停的耐心值。验证 loss 连续多少轮不改善就停止。
            太小 → 训练太早停；太大 → 相当于没有早停。
            一般设 20~50。
    
    返回：
        train_losses : 每轮训练 loss
        val_losses   : 每轮验证 loss
        best_epoch   : 实际最优的轮数（早停记录）
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    # 早停相关变量
    best_val_loss = float('inf')
    best_weights  = None          # 保存最好那一轮的权重
    no_improve    = 0             # 连续多少轮没有改善
    best_epoch    = 0

    n = len(X_train)

    for epoch in range(epochs):
        # ---------- 训练阶段（Mini-batch）----------
        model.train()                           # 开启训练模式（Dropout 生效）
        idx = torch.randperm(n)                 # 随机打乱训练集顺序
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, n, batch_size):   # 每次取 batch_size 个样本
            end = min(start + batch_size, n)
            batch_X = X_train[idx[start:end]]
            batch_y = y_train[idx[start:end]]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        # ---------- 验证阶段 ----------
        model.eval()                            # 关闭 Dropout（预测模式）
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

        # ---------- 早停逻辑 ----------
        if val_loss < best_val_loss - 1e-6:     # 改善了
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            best_epoch    = epoch + 1
        else:                                    # 没改善
            no_improve += 1
            if no_improve >= patience:
                print(f"  早停触发！在第 {best_epoch} 轮时达到最优，共训练 {epoch+1} 轮")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1:>3}] 训练Loss: {avg_train_loss:.5f}  验证Loss: {val_loss:.5f}")

    # 恢复到最优权重
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"  已恢复到第 {best_epoch} 轮的最优权重")

    return train_losses, val_losses, best_epoch


# ============================================================
# 第四部分：评估函数
# ============================================================

def evaluate(model, X_test, y_test_scaled, scaler_disp):
    """
    反归一化后计算真实误差。
    
    返回：
        metrics : dict，包含 RMSE、MAE、R2
        y_real  : 真实位移（mm）
        y_pred  : 预测位移（mm）
    """
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy().flatten()

    y_real = scaler_disp.inverse_transform(y_test_scaled.numpy().reshape(-1, 1)).flatten()
    y_pred = scaler_disp.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_real, y_pred


# ============================================================
# 第五部分：主流程 —— 对比实验
# ============================================================

if __name__ == '__main__':

    CSV_PATH   = 'D:/python-lstm-learning/02_data_processing/landslide_data.csv'
    TIME_STEPS = 10

    print("=" * 55)
    print("  LSTM 调参与优化 —— 对比实验")
    print("=" * 55)

    # ---- 加载数据 ----
    X_train, y_train, X_test, y_test, scaler_disp, df = \
        load_and_prepare_data(CSV_PATH, TIME_STEPS)

    print(f"\n数据信息：")
    print(f"  训练集: {X_train.shape}  →  {y_train.shape}")
    print(f"  测试集: {X_test.shape}   →  {y_test.shape}")

    # ============================================================
    # 实验 1：基础版 LSTM（无优化）
    # ============================================================
    print("\n" + "─" * 45)
    print("实验 1：基础版 LSTM（无 Dropout，无早停）")
    print("─" * 45)

    model_basic = LSTMBasic()
    train_loss_basic, val_loss_basic, _ = train_model(
        model_basic, X_train, y_train, X_test, y_test,
        epochs=200,
        lr=0.001,
        batch_size=len(X_train),   # 用全部数据，不分批（相当于没有 Mini-batch）
        patience=200               # patience = epochs，相当于没有早停
    )
    metrics_basic, y_real, y_pred_basic = evaluate(model_basic, X_test, y_test, scaler_disp)

    print(f"\n  测试结果：")
    print(f"    RMSE = {metrics_basic['RMSE']:.3f} mm")
    print(f"    MAE  = {metrics_basic['MAE']:.3f} mm")
    print(f"    R2   = {metrics_basic['R2']:.4f}")

    # ============================================================
    # 实验 2：优化版 LSTM（Dropout + 早停 + Mini-batch）
    # ============================================================
    print("\n" + "─" * 45)
    print("实验 2：优化版 LSTM（Dropout + 早停 + Mini-batch）")
    print("─" * 45)

    model_opt = LSTMOptimized(dropout=0.2)
    train_loss_opt, val_loss_opt, best_ep = train_model(
        model_opt, X_train, y_train, X_test, y_test,
        epochs=200,
        lr=0.001,
        batch_size=16,    # Mini-batch：每次用 16 个样本
        patience=30       # 早停：连续 30 轮不改善就停
    )
    metrics_opt, _, y_pred_opt = evaluate(model_opt, X_test, y_test, scaler_disp)

    print(f"\n  测试结果：")
    print(f"    RMSE = {metrics_opt['RMSE']:.3f} mm")
    print(f"    MAE  = {metrics_opt['MAE']:.3f} mm")
    print(f"    R2   = {metrics_opt['R2']:.4f}")

    # ============================================================
    # 实验 3：不同超参数对比（time_steps = 5 vs 10 vs 20）
    # ============================================================
    print("\n" + "─" * 45)
    print("实验 3：time_steps 对结果的影响")
    print("─" * 45)

    results_ts = {}
    for ts in [5, 10, 20]:
        Xtr, ytr, Xte, yte, sc, _ = load_and_prepare_data(CSV_PATH, ts)
        m = LSTMOptimized(dropout=0.2)
        train_model(m, Xtr, ytr, Xte, yte,
                    epochs=200, lr=0.001, batch_size=16, patience=30)
        metrics, _, _ = evaluate(m, Xte, yte, sc)
        results_ts[ts] = metrics
        print(f"  time_steps={ts:>2}:  RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  R2={metrics['R2']:.4f}")

    # ============================================================
    # 画图
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LSTM 调参与优化 —— 对比实验', fontsize=15, fontweight='bold')

    # 子图1：训练/验证 Loss 曲线对比
    ax = axes[0, 0]
    ax.plot(train_loss_basic, label='基础版 - 训练Loss', color='steelblue', linewidth=1.5)
    ax.plot(val_loss_basic,   label='基础版 - 验证Loss', color='steelblue', linewidth=1.5, linestyle='--')
    ax.plot(train_loss_opt,   label='优化版 - 训练Loss', color='tomato', linewidth=1.5)
    ax.plot(val_loss_opt,     label='优化版 - 验证Loss', color='tomato', linewidth=1.5, linestyle='--')
    ax.axvline(best_ep - 1, color='gray', linestyle=':', linewidth=1.2, label=f'早停点(第{best_ep}轮)')
    ax.set_title('Loss 曲线对比')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 子图2：基础版预测结果
    ax = axes[0, 1]
    x_axis = range(len(y_real))
    ax.plot(x_axis, y_real,        label='真实值', color='black', linewidth=2)
    ax.plot(x_axis, y_pred_basic,  label=f'基础版预测 (R2={metrics_basic["R2"]:.3f})',
            color='steelblue', linewidth=1.5, linestyle='--')
    ax.plot(x_axis, y_pred_opt,    label=f'优化版预测 (R2={metrics_opt["R2"]:.3f})',
            color='tomato', linewidth=1.5, linestyle='--')
    ax.set_title('预测值 vs 真实值（测试集）')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('累计位移 (mm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 子图3：time_steps 对比柱状图
    ax = axes[1, 0]
    ts_labels = [f'ts={ts}' for ts in results_ts.keys()]
    rmse_vals  = [results_ts[ts]['RMSE'] for ts in results_ts]
    r2_vals    = [results_ts[ts]['R2']   for ts in results_ts]

    x = np.arange(len(ts_labels))
    bars = ax.bar(x - 0.2, rmse_vals, width=0.35, label='RMSE (mm)', color='steelblue', alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, r2_vals, 'o-', color='tomato', linewidth=2, label='R2', markersize=8)
    ax.set_title('不同 time_steps 的效果对比')
    ax.set_xticks(x)
    ax.set_xticklabels(ts_labels)
    ax.set_ylabel('RMSE (mm)', color='steelblue')
    ax2.set_ylabel('R2', color='tomato')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 子图4：调参总结表格
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['参数/技巧', '作用', '推荐值'],
        ['hidden_size', 'LSTM记忆单元数', '32~128'],
        ['num_layers', 'LSTM层数', '1~3'],
        ['time_steps', '看多长历史', '5~30'],
        ['learning_rate', '步长', '1e-3~1e-4'],
        ['Dropout', '防过拟合', '0.1~0.3'],
        ['batch_size', '每批样本数', '8~32'],
        ['patience', '早停耐心值', '20~50'],
    ]
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#D9E1F2')
    ax.set_title('LSTM 常用超参数速查', pad=10)

    plt.tight_layout()
    out_path = 'D:/python-lstm-learning/05_deep_learning/03_lstm_optimization_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存：{out_path}")

    # ============================================================
    # 打印对比总结
    # ============================================================
    print("\n" + "=" * 55)
    print("  最终对比总结")
    print("=" * 55)
    print(f"  {'模型':<20} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'基础版(无优化)':<20} {metrics_basic['RMSE']:>8.3f} {metrics_basic['MAE']:>8.3f} {metrics_basic['R2']:>8.4f}")
    print(f"  {'优化版(+Dropout+早停)':<20} {metrics_opt['RMSE']:>8.3f} {metrics_opt['MAE']:>8.3f} {metrics_opt['R2']:>8.4f}")
    print()
    print("  ✅ 本节课学到的技巧：")
    print("     1. Dropout — 随机关掉神经元，防止过拟合")
    print("     2. 早停   — 自动在最优轮停下，不白费时间")
    print("     3. Mini-batch — 分批训练，更稳定更快")
    print("     4. 对比实验   — 调参要有对照，不能凭感觉")
    print()
    print("  下一课：06_project — 用真实白水河数据跑完整毕设流程")
