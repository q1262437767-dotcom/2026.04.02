# ============================================================
# Pandas 基础 - 表格数据处理神器
# 学习目标：掌握 DataFrame 读写、数据筛选、缺失值处理
# ============================================================

import pandas as pd
import numpy as np

print("=" * 50)
print("Pandas 基础 - 表格数据处理")
print("=" * 50)


# ─────────────────────────────────────────────
# 【1. DataFrame 是什么？】
# ─────────────────────────────────────────────
print("\n【1. DataFrame —— 带标签的表格】")
print("-" * 40)

# DataFrame 就是一个"Excel表格"在 Python 里的样子
data = {
    '日期': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05'],
    '降雨量': [45.2, 128.6, 89.3, 210.5, 67.8],
    '水位':   [155.2, 162.8, 168.5, 172.1, 158.3],
    '位移':   [2.1, 5.8, 8.3, 15.6, 10.2]
}

df = pd.DataFrame(data)
print(df)
print(f"\n类型: {type(df)}")
print(f"形状: {df.shape}  → {df.shape[0]}行 × {df.shape[1]}列")

# .describe() 一键统计摘要
print("\n--- 数据统计摘要 ---")
print(df.describe())

# .info() 查看数据概况
print("\n--- 数据类型信息 ---")
df.info()


# ─────────────────────────────────────────────
# 【2. 读取真实数据（对应毕设）】
# ─────────────────────────────────────────────
print("\n\n【2. 读取和保存数据】")
print("-" * 40)

# 2.1 读取 CSV
print("--- 读取 CSV ---")
csv_path = 'landslide_data.csv'
sample_df = pd.DataFrame({
    'date': pd.date_range('2020-01', periods=36, freq='ME'),
    'rainfall': np.random.uniform(20, 200, 36).round(1),
    'water_level': np.random.uniform(145, 175, 36).round(1),
    'displacement': np.cumsum(np.random.uniform(0.5, 3.0, 36)).round(1)
})
sample_df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"已生成示例CSV: {csv_path}")

df_read = pd.read_csv(csv_path)
print(df_read.head())  # head() 默认显示前5行

# 2.2 读取 Excel（你毕设用的就是这个！）
print("\n--- 读取 Excel（毕设常用） ---")
print("用法: df = pd.read_excel('数据.xlsx', sheet_name='Sheet1')")
print("      df.to_excel('结果.xlsx', index=False)")
print("注意: 需要安装 openpyxl:  pip install openpyxl")

# 2.3 常用读取参数
print("\n--- 常用参数 ---")
print("pd.read_csv('file.csv', encoding='gbk')   # 中文乱码用gbk")
print("pd.read_csv('file.csv', skiprows=1)        # 跳过前1行")
print("pd.read_csv('file.csv', nrows=100)         # 只读前100行")
print("pd.read_csv('file.csv', na_values='NA')    # 指定哪些值算缺失")


# ─────────────────────────────────────────────
# 【3. 数据筛选 —— 取你想要的行列】
# ─────────────────────────────────────────────
print("\n\n【3. 数据筛选】")
print("-" * 40)

df = sample_df.copy()  # 用刚才生成的数据继续练习

# 3.1 取列
print("--- 取单列（返回Series） ---")
print(df['rainfall'].head())

print("\n--- 取多列（返回DataFrame） ---")
print(df[['date', 'displacement']].head())

# 3.2 取行 —— iloc vs loc
print("\n--- iloc: 按位置索引（从0开始） ---")
print(df.iloc[0])          # 第1行
print(df.iloc[2:5])        # 第3~5行（左闭右开，跟列表切片一样）
print(df.iloc[:, 1])       # 所有行的第2列（rainfall）

print("\n--- loc: 按标签索引（用列名/行名） ---")
print(df.loc[0])           # 标签为0的行
print(df.loc[0:3])         # 标签0到3的行（注意：loc 是闭区间！跟iloc不同）
print(df.loc[:, 'rainfall'])  # 所有行的 rainfall 列

# 3.3 条件筛选
print("\n--- 条件筛选 ---")
high_rain = df[df['rainfall'] > 150]
print(f"降雨量 > 150mm 的月份（共{len(high_rain)}个）:")
print(high_rain[['date', 'rainfall']])

# 多条件：用 &（且）、|（或），每个条件要加括号！
high_rain_wet = df[(df['rainfall'] > 150) & (df['water_level'] > 170)]
print(f"\n降雨>150 且 水位>170 的月份:")
print(high_rain_wet[['date', 'rainfall', 'water_level']])


# ─────────────────────────────────────────────
# 【4. 缺失值处理 —— 真实数据经常有"空"】
# ─────────────────────────────────────────────
print("\n\n【4. 缺失值处理】")
print("-" * 40)

# 模拟有缺失值的数据
df_missing = pd.DataFrame({
    'month': ['1月', '2月', '3月', '4月', '5月', '6月'],
    'rainfall': [45, np.nan, 128, 89, np.nan, 210],
    'displacement': [2.1, 5.8, np.nan, 8.3, 15.6, np.nan]
})
print("原始数据（有NaN）:")
print(df_missing)

# 4.1 检查缺失值
print(f"\n每列缺失值数量:")
print(df_missing.isnull().sum())
print(f"\n总缺失值: {df_missing.isnull().sum().sum()}")

# 4.2 删除缺失值
print("\n--- dropna: 删除有缺失的行 ---")
print(df_missing.dropna())  # 任何一列有NaN就删掉整行

# 4.3 填充缺失值
print("\n--- fillna: 用指定值填充 ---")
print(df_missing.fillna(0))  # 全部填0（不一定合理）

print("\n--- 用均值填充 ---")
rain_mean = df_missing['rainfall'].mean()
print(f"降雨量均值: {rain_mean:.1f}")
df_filled = df_missing.copy()
df_filled['rainfall'] = df_filled['rainfall'].fillna(rain_mean)
print(df_filled)

print("\n--- 用前一个值填充（常用！） ---")
print(df_missing.ffill())

# TODO: 你的毕设里如果有缺失的GPS数据，应该用哪种方式填充？
# 提示：滑坡数据是连续的，用前一个值填充通常比均值更合理

# ffill()

# ─────────────────────────────────────────────
# 【5. 新增/修改列 —— 数据加工】
# ─────────────────────────────────────────────
print("\n\n【5. 新增和修改列】")
print("-" * 40)

df = sample_df.copy()

# 5.1 新增列（直接赋值）
df['year'] = df['date'].dt.year       # 提取年份
df['month'] = df['date'].dt.month     # 提取月份
print("新增 year 和 month 列:")
print(df[['date', 'year', 'month']].head())

# 5.2 基于现有列计算新列
df['位移增量'] = df['displacement'].diff()  # diff() = 当前值 - 上一行的值
print("\n新增 位移增量 列（diff计算相邻差值）:")
print(df[['date', 'displacement', '位移增量']].head(8))

# 5.3 条件赋值（分类）
df['预警等级'] = '正常'
df.loc[df['displacement'] > 30, '预警等级'] = '黄色'
df.loc[df['displacement'] > 50, '预警等级'] = '橙色'
df.loc[df['displacement'] > 80, '预警等级'] = '红色'
print("\n根据位移设置预警等级:")
print(df[['date', 'displacement', '预警等级']].head(10))

# 5.4 删除列
df_dropped = df.drop(columns=['year', 'month'])
print(f"\n删除 year, month 后: {df_dropped.shape}")


# ─────────────────────────────────────────────
# 【6. 滑动窗口 —— LSTM 的核心数据构造方法】
# ─────────────────────────────────────────────
print("\n\n【6. 滑动窗口构造（LSTM 必备！）】")
print("-" * 40)

# 这就是你毕设里构造训练数据的核心逻辑！
print("用过去3个月的数据，预测下1个月的位移")
print("例如: 用 [1月, 2月, 3月] 的数据 → 预测 4月")

# 简单演示
series = np.array([2.1, 5.8, 8.3, 15.6, 10.2, 12.7, 18.3, 22.1, 25.6, 28.9])

window_size = 3  # 用前3个预测下一个

X = []  # 特征（输入）
y = []  # 标签（输出）

for i in range(len(series) - window_size):
    X.append(series[i : i + window_size])    # 取3个作为输入
    y.append(series[i + window_size])         # 第4个作为目标

X = np.array(X)
y = np.array(y)

print(f"\n输入 X 形状: {X.shape}  → {X.shape[0]}个样本，每个看{X.shape[1]}个月")
print(f"输出 y 形状: {y.shape}  → {y.shape[0]}个预测值")

print("\n逐个展示:")
for i in range(len(X)):
    print(f"  样本{i+1}: 输入 {X[i]} → 预测 {y[i]:.1f}")

print("\n提示: 你的毕设里不是只看位移一个特征，")
print("而是同时看 [降雨, 水位, 位移] 三个特征，")
print("所以 X 的形状是 (样本数, 3, 3) 而不是 (样本数, 3, 1)")


# ─────────────────────────────────────────────
# 【7. 分组统计 —— 按类别汇总】
# ─────────────────────────────────────────────
print("\n\n【7. 分组统计】")
print("-" * 40)

df = sample_df.copy()
df['year'] = df['date'].dt.year

# groupby + 聚合
yearly_stats = df.groupby('year').agg({
    'rainfall': ['mean', 'max', 'sum'],
    'water_level': ['mean', 'max'],
    'displacement': ['mean', 'max', 'min']
}).round(1)

print("各年份统计汇总:")
print(yearly_stats)

print("\n--- 其他常用聚合 ---")
print("df.groupby('year')['displacement'].max()       # 每年最大位移")
print("df.groupby('year').size()                       # 每年数据条数")
print("df.groupby('year').mean(numeric_only=True)      # 每年所有数值列的均值")


# ─────────────────────────────────────────────
# 【8. 排序和去重】
# ─────────────────────────────────────────────
print("\n\n【8. 排序和去重】")
print("-" * 40)

df = pd.DataFrame({
    '监测点': ['GNSS-01', 'GNSS-03', 'GNSS-02', 'GNSS-05', 'GNSS-04'],
    '累计位移': [45.2, 128.6, 89.3, 15.6, 67.8]
})

print("原始数据:")
print(df)

print("\n--- 按位移从大到小排 ---")
print(df.sort_values('累计位移', ascending=False))

print("\n--- 按监测点名称排 ---")
print(df.sort_values('监测点'))

print("\n--- 去重 ---")
df_dup = pd.concat([df, df.iloc[[0]]])  # 复制第一行制造重复
print(f"去重前: {len(df_dup)}行")
print(f"去重后: {len(df_dup.drop_duplicates())}行")


# ============================================================
# TODO 练习
# ============================================================
print("\n\n" + "=" * 50)
print("TODO 练习题")
print("=" * 50)

# TODO 1: 从 sample_df 中筛选出水位高于均值的月份，并按降雨量降序排列
# 提示: df[df['water_level'] > df['water_level'].mean()].sort_values('rainfall', ascending=False)

# ① 算出水位均值
mean_water = sample_df['water_level'].mean()

# ② 筛选水位高于均值的行
filtered = sample_df[sample_df['water_level'] > mean_water]

# ③ 按降雨量降序排列（ascending=False 就是降序）
result = filtered.sort_values('rainfall', ascending=False)

print(result[['date', 'rainfall', 'water_level']])

# TODO 2: 计算每季度的平均降雨量（3个月一组）
# 提示: 可以用 groupby 或者 每3行取均值

# Pandas 方式：resample 按季度重采样
quarterly = sample_df.set_index('date')['rainfall'].resample('QE').mean()
print(quarterly)
# Numpy方法：把降雨量变成数组，每3个月一组求均值
rain = sample_df['rainfall'].values  # 转成 NumPy 数组
quarterly_avg = rain.reshape(-1, 3).mean(axis=1)


# TODO 3: 找到位移增长最快的连续3个月
# 提示: 用 diff() 计算增量，再找最大的连续增长
# ① 算每月位移增量
sample_df['增量'] = sample_df['displacement'].diff()

# ② 用 rolling(3).sum() 算"连续3个月的总增量"
sample_df['3月总增量'] = sample_df['增量'].rolling(window=3).sum()

# ③ 找到最大值所在的行
max_idx = sample_df['3月总增量'].idxmax()

# ④ 输出结果：那3个月就是 max_idx-2, max_idx-1, max_idx
print("位移增长最快的连续3个月:")
for i in range(int(max_idx) - 2, int(max_idx) + 1):
    print(f"  {sample_df.loc[i, 'date']}  位移: {sample_df.loc[i, 'displacement']:.1f}  "
          f"增量: {sample_df.loc[i, '增量']:.1f}")
print(f"3个月总增长: {sample_df.loc[max_idx, '3月总增量']:.1f} mm")
