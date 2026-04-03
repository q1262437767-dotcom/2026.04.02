"""
第二课：条件语句与循环
=====================
让程序根据条件做不同的事，重复执行任务

运行方式: python 02_conditions_loops.py
"""

print("=" * 50)
print("第二课：条件语句与循环")
print("=" * 50)

# ─────────────────────────────────────
# 1. if-elif-else 条件判断
# ─────────────────────────────────────
print("\n【1. 条件判断】")

displacement = 45.2  # 位移量 mm

# 根据位移量判断滑坡预警等级
if displacement < 10:
    level = "安全（蓝色）"
elif displacement < 30:
    level = "注意（黄色）"
elif displacement < 60:
    level = "预警（橙色）"
else:
    level = "危险（红色）"

print(f"位移量: {displacement} mm → 预警等级: {level}")

# 多条件组合
rainfall = 85  # mm/天
reservoir_level = 175.5  # m（库水位）

if rainfall > 80 and reservoir_level > 175:
    print("⚠️ 暴雨 + 高水位，需加强监测！")
elif rainfall > 80 or reservoir_level > 175:
    print("⚡ 单一因素超限，保持关注")
else:
    print("✅ 各项指标正常")


# ─────────────────────────────────────
# 2. for 循环 —— 遍历序列
# ─────────────────────────────────────
print("\n【2. for 循环】")

# 遍历列表
sensors = ["GNSS-01", "GNSS-02", "GNSS-03", "GNSS-04"]
print("监测点列表:")
for sensor in sensors:
    print(f"  - {sensor}")

# range() —— 生成数字序列
print("\n1到5的平方:")
for i in range(1, 6):
    print(f"  {i}² = {i ** 2}")

# enumerate() —— 同时获取索引和值
print("\n带编号遍历:")
for idx, sensor in enumerate(sensors):
    print(f"  第{idx + 1}个: {sensor}")

# 模拟一周监测数据
print("\n一周位移数据（模拟）:")
import random
random.seed(42)  # 固定随机种子，每次结果一样
weekly_data = []
for day in range(1, 8):
    disp = random.uniform(2.0, 15.0)  # 随机位移 2~15mm
    weekly_data.append(round(disp, 2))
    print(f"  第{day}天: {disp:.2f} mm")


# ─────────────────────────────────────
# 3. while 循环 —— 条件满足就一直执行
# ─────────────────────────────────────
print("\n【3. while 循环】")

# 模拟库水位上升过程
water_level = 155.0  # 初始水位
target = 175.0       # 汛限水位
day = 0

print(f"初始水位: {water_level}m，目标: {target}m")
while water_level < target:
    day += 1
    water_level += random.uniform(0.5, 2.0)
    water_level = min(water_level, target)
    bar = "█" * int(water_level / target * 30)
    print(f"  第{day:2d}天: {water_level:.1f}m {bar}")

print(f"  🎯 第{day}天达到汛限水位 {target}m")


# ─────────────────────────────────────
# 4. break 和 continue —— 控制循环
# ─────────────────────────────────────
print("\n【4. break 和 continue】")

# break: 找到第一个超过阈值的数据就停止
print("查找首个超过 10mm 的位移:")
displacements = [3.2, 5.1, 7.8, 12.3, 8.9, 15.6, 4.2]
for i, d in enumerate(displacements):
    if d > 10:
        print(f"  第{i + 1}天，位移 {d}mm 超过阈值！停止监测。")
        break

# continue: 跳过不合格的数据
print("\n过滤位移数据（跳过异常值 >20mm）:")
for d in displacements:
    if d > 20:
        continue  # 跳过这轮，继续下一个
    print(f"  有效数据: {d}mm")


# ─────────────────────────────────────
# 5. 列表推导式 —— Python 特色写法
# ─────────────────────────────────────
print("\n【5. 列表推导式】")

# 普通写法
squares_normal = []
for i in range(1, 6):
    squares_normal.append(i ** 2)

# 推导式写法（一行搞定，更 Pythonic）
squares = [i ** 2 for i in range(1, 6)]
print(f"平方数: {squares}")

# 带条件的推导式
# 只保留位移超过 5mm 的数据
large_disp = [d for d in displacements if d > 5]
print(f"位移 > 5mm: {large_disp}")

# 转换单位 mm → cm
disp_cm = [round(d / 10, 2) for d in displacements]
print(f"位移转cm: {disp_cm}")


# ─────────────────────────────────────
# ✏️ 动手练习
# ─────────────────────────────────────
print("\n" + "=" * 50)
print("✏️ 动手练习")
print("=" * 50)

# 练习1：统计一周中预警天数
print(f"\n一周数据: {weekly_data}")
warning_days = [d for d in weekly_data if d > 10]
print(f"预警天数(>10mm): {len(warning_days)} 天")
print(f"预警数据: {warning_days}")

# 练习2：计算累计位移
cumulative = 0
print("\n累计位移:")
for i, d in enumerate(weekly_data):
    cumulative += d
    print(f"  第{i + 1}天: 今日 {d:.2f}mm，累计 {cumulative:.2f}mm")

# 练习3：你来试试
# TODO: 用 for 循环打印 1~20 中所有能被3整除的数
# TODO: 用 while 循环计算 1+2+3+...+100 的和

print("\n✅ 第二课完成！下一课：函数与列表")
