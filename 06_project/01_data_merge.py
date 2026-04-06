"""
白水河滑坡数据合并预处理
========================
将三个原始数据源合并为一个月度CSV文件：
- 降雨量(Sheet1): 逐日 → 按月求和
- 长江水位(Sheet2): 逐日 → 按月取均值
- GPS位移(6个Sheet): 每年一个 → ZG91月位移增量 → 累计

注意: GPS表中Row12的时间标签"X.1"实际为10月（xls截断）
输出: baishuihe_monthly.csv
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 路径设置
# ============================================================
BASE_DIR = r"D:/Study/毕设/2007-2012年长江三峡库区秭归县白水河滑坡基本特征及监测数据"
RAINFALL_FILE = os.path.join(BASE_DIR, "白水河滑坡降雨量、长江水位观测数据资料表（2007-2012年）.xls")
GPS_FILE = os.path.join(BASE_DIR, "白水河滑坡地表位移GPS监测成果表（2007-2012年）.xls")
OUTPUT_DIR = r"D:/python-lstm-learning/06_project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取逐日降雨量（Sheet1）
# ============================================================
print("=" * 60)
print("【步骤1】读取逐日降雨量")
df_rain = pd.read_excel(RAINFALL_FILE, sheet_name=0)
df_rain.columns = ["date", "rainfall", "col2", "col3"]
df_rain = df_rain[["date", "rainfall"]]
df_rain["date"] = pd.to_datetime(df_rain["date"])
df_rain["rainfall"] = df_rain["rainfall"].fillna(0)
print(f"  记录: {len(df_rain)} 条, 范围: {df_rain['date'].iloc[0].date()} ~ {df_rain['date'].iloc[-1].date()}")

# ============================================================
# 2. 读取逐日长江水位（Sheet2）
# ============================================================
print("\n【步骤2】读取逐日长江水位")
df_water = pd.read_excel(RAINFALL_FILE, sheet_name=1)
df_water.columns = ["date", "water_level", "col2", "col3"]
df_water = df_water[["date", "water_level"]]
df_water["date"] = pd.to_datetime(df_water["date"])
print(f"  记录: {len(df_water)} 条, 范围: {df_water['date'].iloc[0].date()} ~ {df_water['date'].iloc[-1].date()}")

# ============================================================
# 3. 逐日 → 月度汇总
# ============================================================
print("\n【步骤3】逐日数据按月汇总")
df_daily = pd.merge(df_rain, df_water, on="date", how="inner")
df_daily["year_month"] = df_daily["date"].dt.to_period("M")

monthly_climate = df_daily.groupby("year_month").agg(
    rainfall_mm=("rainfall", "sum"),
    water_level_m=("water_level", "mean"),
).reset_index()
monthly_climate["date"] = monthly_climate["year_month"].dt.to_timestamp()
print(f"  月度气候数据: {len(monthly_climate)} 个月")

# ============================================================
# 4. 读取GPS位移数据
# ============================================================
print("\n【步骤4】读取GPS位移数据（ZG91/ZG67）")
xls = pd.ExcelFile(GPS_FILE)

gps_all = []

for sheet_name in xls.sheet_names:
    year = int(sheet_name.replace("年", ""))
    df = pd.read_excel(GPS_FILE, sheet_name=sheet_name, header=None)

    # 每年数据: Row2~Row13, 共12行
    # Row2 = 上一年12月, Row3~Row12 = 1~9月, Row12 = 10月, Row13 = 11月
    # 注意: Row12的时间标签是 "X.1"（实际是10月，xls截断）

    row_month_map = {
        2: None,   # 上一年12月, 特殊处理
        3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9,
        12: 10,    # "X.1" 实际是10月
        13: 11,
    }

    for row_idx, month in row_month_map.items():
        if row_idx >= len(df):
            continue

        row = df.iloc[row_idx]

        if month is None:
            # Row2 = 上一年12月
            actual_year = year - 1
            actual_month = 12
        else:
            actual_year = year
            actual_month = month

        try:
            delta_f = float(row.iloc[3])
            if np.isnan(delta_f):
                continue  # 跳过缺失数据（如2010年3月）
            gps_all.append({
                "year": actual_year,
                "month": actual_month,
                "delta_f": delta_f,
            })
        except (ValueError, TypeError):
            pass

df_gps = pd.DataFrame(gps_all)

# 去重（如果有的话）
df_gps = df_gps.drop_duplicates(subset=["year", "month"]).sort_values(["year", "month"]).reset_index(drop=True)

# 创建日期列
df_gps["date"] = pd.to_datetime(
    df_gps["year"].astype(int).astype(str) + "-" +
    df_gps["month"].astype(int).astype(str).str.zfill(2) + "-01"
)

# 累计位移
df_gps["cum_displacement_mm"] = df_gps["delta_f"].cumsum()

print(f"  GPS月度数据: {len(df_gps)} 条")
print(f"  范围: {df_gps['date'].iloc[0].date()} ~ {df_gps['date'].iloc[-1].date()}")
print(f"  月位移增量: {df_gps['delta_f'].min():.1f} ~ {df_gps['delta_f'].max():.1f} mm")
print(f"  累计位移末值: {df_gps['cum_displacement_mm'].iloc[-1]:.1f} mm")

# ============================================================
# 5. 三表合并
# ============================================================
print("\n【步骤5】三表合并")
df_final = pd.merge(
    monthly_climate[["date", "rainfall_mm", "water_level_m"]],
    df_gps[["date", "delta_f", "cum_displacement_mm"]],
    on="date",
    how="inner"
)
df_final = df_final.rename(columns={"delta_f": "displacement_mm"})
df_final = df_final.sort_values("date").reset_index(drop=True)
df_final = df_final.dropna()

print(f"  最终数据: {len(df_final)} 条月度记录")
print(f"  范围: {df_final['date'].iloc[0].date()} ~ {df_final['date'].iloc[-1].date()}")
print()

# ============================================================
# 6. 保存
# ============================================================
output_file = os.path.join(OUTPUT_DIR, "baishuihe_monthly.csv")
df_final.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"已保存: {output_file}")
print()
print("数据预览:")
print(df_final.to_string(index=False))
print()
print("描述统计:")
print(df_final.describe().round(2).to_string())
