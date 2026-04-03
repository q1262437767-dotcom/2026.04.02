"""
第三课：函数与数据结构
=====================
函数让你复用代码，列表/字典让你组织数据

运行方式: python 03_functions.py
"""

print("=" * 50)
print("第三课：函数与数据结构")
print("=" * 50)

# ─────────────────────────────────────
# 1. 定义函数 —— 把代码封装成可复用的块
# ─────────────────────────────────────
print("\n【1. 函数基础】")

# 最简单的函数
def greet():
    print("你好！欢迎学习 Python！")

greet()  # 调用函数


# 带参数的函数
def calc_rate(displacement, days):
    """计算平均位移速率"""
    return displacement / days

rate = calc_rate(68.5, 3)
print(f"平均位移速率: {rate:.2f} mm/天")


# 带默认参数
def get_warning_level(displacement, threshold=10.0):
    """判断预警等级"""
    if displacement < threshold:
        return "正常"
    elif displacement < threshold * 3:
        return "预警"
    else:
        return "危险"

print(get_warning_level(5.2))       # 正常（用默认阈值10）
print(get_warning_level(25.0, 20))   # 危险（自定义阈值20）


# ─────────────────────────────────────
# 2. 函数进阶 —— 多返回值、lambda
# ─────────────────────────────────────
print("\n【2. 函数进阶】")

# 多返回值（返回元组）
def analyze_data(data):
    """返回最大值、最小值、平均值"""
    return max(data), min(data), sum(data) / len(data)

displacements = [3.2, 5.1, 12.3, 8.9, 15.6, 4.2]
mx, mn, avg = analyze_data(displacements)
print(f"数据: {displacements}")
print(f"最大: {mx}mm, 最小: {mn}mm, 平均: {avg:.2f}mm")

# lambda 匿名函数（一行搞定简单逻辑）
square = lambda x: x ** 2
double = lambda x: x * 2

print(f"square(5) = {square(5)}")
print(f"double(3) = {double(3)}")

# 配合 map 使用：批量转换
mm_values = [10, 25, 38, 42]
cm_values = list(map(lambda x: x / 10, mm_values))
print(f"mm转cm: {cm_values}")


# ─────────────────────────────────────
# 3. 列表 (list) —— 最常用的数据结构
# ─────────────────────────────────────
print("\n【3. 列表操作】")

# 创建列表
data = [3.2, 5.1, 12.3, 8.9, 15.6, 4.2]
print(f"原始数据: {data}")

# 增
data.append(7.8)              # 末尾添加
data.insert(0, 1.1)           # 在第0个位置插入
print(f"添加后:   {data}")

# 删
data.pop()                     # 删除最后一个
data.remove(12.3)              # 删除值为12.3的元素
print(f"删除后:   {data}")

# 改
data[0] = 2.5
print(f"修改后:   {data}")

# 查
print(f"第一个: {data[0]}")
print(f"最后一个: {data[-1]}")
print(f"前3个: {data[:3]}")
print(f"长度: {len(data)}")

# 排序
data_sorted = sorted(data)
print(f"升序: {data_sorted}")
print(f"降序: {sorted(data, reverse=True)}")


# ─────────────────────────────────────
# 4. 字典 (dict) —— 键值对存储
# ─────────────────────────────────────
print("\n【4. 字典操作】")

# 创建字典
sensor = {
    "name": "GNSS-01",
    "location": "黄土坡",
    "displacement": 23.5,
    "unit": "mm",
    "active": True
}

# 访问
print(f"传感器名: {sensor['name']}")
print(f"位移值: {sensor['displacement']} {sensor['unit']}")

# 修改
sensor["displacement"] = 25.8
sensor["warning_level"] = "预警"
print(f"更新后: {sensor}")

# 遍历
print("\n遍历传感器信息:")
for key, value in sensor.items():
    print(f"  {key}: {value}")

# 多个传感器的列表
sensors = [
    {"name": "GNSS-01", "displacement": 23.5},
    {"name": "GNSS-02", "displacement": 45.2},
    {"name": "GNSS-03", "displacement": 8.1},
    {"name": "GNSS-04", "displacement": 67.8},
]

# 找出位移最大的传感器
max_sensor = max(sensors, key=lambda s: s["displacement"])
print(f"\n位移最大: {max_sensor['name']} = {max_sensor['displacement']}mm")

# 筛选预警传感器
warning = [s for s in sensors if s["displacement"] > 30]
print(f"预警传感器: {[s['name'] for s in warning]}")


# ─────────────────────────────────────
# 5. 综合实战：模拟监测数据处理
# ─────────────────────────────────────
print("\n【5. 综合实战：监测数据统计】")

import random
random.seed(42)

def generate_daily_data(days=30):
    """模拟生成每天监测数据"""
    data = []
    for day in range(1, days + 1):
        rainfall = round(random.uniform(0, 120), 1)
        water_level = round(random.uniform(155, 178), 1)
        displacement = round(random.uniform(1, 25), 2)
        data.append({
            "day": day,
            "rainfall": rainfall,
            "water_level": water_level,
            "displacement": displacement,
            "warning": "危险" if displacement > 15 else ("预警" if displacement > 8 else "正常")
        })
    return data

# 生成数据
monthly_data = generate_daily_data(30)

# 统计
normal_days = sum(1 for d in monthly_data if d["warning"] == "正常")
warning_days = sum(1 for d in monthly_data if d["warning"] == "预警")
danger_days = sum(1 for d in monthly_data if d["warning"] == "危险")
total_disp = sum(d["displacement"] for d in monthly_data)
avg_disp = total_disp / len(monthly_data)
max_rain = max(monthly_data, key=lambda d: d["rainfall"])

print(f"  统计周期: 30天")
print(f"  正常天数: {normal_days}天")
print(f"  预警天数: {warning_days}天")
print(f"  危险天数: {danger_days}天")
print(f"  累计位移: {total_disp:.2f}mm")
print(f"  平均位移: {avg_disp:.2f}mm/天")
print(f"  最大降雨: 第{max_rain['day']}天, {max_rain['rainfall']}mm")

# 打印前5天数据
print("\n  前5天数据:")
for d in monthly_data[:5]:
    print(f"    第{d['day']:2d}天 | 降雨:{d['rainfall']:5.1f}mm | "
          f"水位:{d['water_level']:5.1f}m | 位移:{d['displacement']:5.2f}mm | {d['warning']}")


# ─────────────────────────────────────
# ✏️ 动手练习
# ─────────────────────────────────────
print("\n" + "=" * 50)
print("✏️ 动手练习")
print("=" * 50)

# 练习1：写一个函数，判断一个数是奇数还是偶数
def odd_or_even(n):
    return "偶数" if n % 2 == 0 else "奇数"

for i in range(1, 6):
    print(f"  {i} 是{odd_or_even(i)}")

# 练习2：从 monthly_data 中找出所有危险天的编号
danger_list = [d["day"] for d in monthly_data if d["warning"] == "危险"]
print(f"\n危险天数编号: {danger_list}")

# 练习3：你来试试
# TODO: 写一个函数 celsius_to_fahrenheit(c)，把摄氏度转为华氏度
# TODO: 用字典存储3门课的成绩，计算平均分

print("\n✅ 第三课完成！下一课：文件读写与异常处理")
