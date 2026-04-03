"""
第一课：变量与数据类型
=====================
Python 中一切皆对象，变量就是给数据贴的标签

运行方式: python 01_variables.py
"""

print("=" * 50)
print("第一课：变量与数据类型")
print("=" * 50)

# ─────────────────────────────────────
# 1. 变量 —— 不需要声明类型，直接赋值
# ─────────────────────────────────────
print("\n【1. 变量赋值】")

name = "小明"           # 字符串
age = 20                # 整数
height = 1.75           # 浮点数
is_student = True       # 布尔值

print(f"姓名: {name}")
print(f"年龄: {age}")
print(f"身高: {height}m")
print(f"是否学生: {is_student}")

# 查看变量的类型
print(f"\nname 的类型: {type(name)}")       # <class 'str'>
print(f"age 的类型: {type(age)}")         # <class 'int'>
print(f"height 的类型: {type(height)}")   # <class 'float'>
print(f"is_student 的类型: {type(is_student)}")  # <class 'bool'>


# ─────────────────────────────────────
# 2. 字符串 (str) —— 文本数据
# ─────────────────────────────────────
print("\n【2. 字符串操作】")

s = "Hello, Python!"

# 常用操作
print(f"原字符串: {s}")
print(f"转大写: {s.upper()}")
print(f"转小写: {s.lower()}")
print(f"长度: {len(s)}")
print(f"替换: {s.replace('Python', 'World')}")
print(f"分割: {s.split(', ')}")
print(f"切片 [0:5]: {s[0:5]}")     # 取第0到第4个字符

# 字符串拼接
first = "滑坡"
second = "位移预测"
print(f"拼接: {first + second}")

# f-string 格式化（推荐用法）
displacement = 23.4567
print(f"今日位移量: {displacement:.2f} mm")  # 保留2位小数


# ─────────────────────────────────────
# 3. 数字 —— 整数和浮点数
# ─────────────────────────────────────
print("\n【3. 数字运算】")

a, b = 10, 3
print(f"a = {a}, b = {b}")
print(f"加法: {a + b}")       # 13
print(f"减法: {a - b}")       # 7
print(f"乘法: {a * b}")       # 30
print(f"除法: {a / b}")       # 3.333...
print(f"整除: {a // b}")      # 3（只取整数部分）
print(f"取余: {a % b}")       # 1
print(f"幂运算: {a ** b}")    # 1000

# 数学函数
import math
print(f"平方根: {math.sqrt(16)}")    # 4.0
print(f"圆周率: {math.pi:.4f}")      # 3.1416


# ─────────────────────────────────────
# 4. 布尔值 (bool) —— True / False
# ─────────────────────────────────────
print("\n【4. 布尔值与比较】")

print(f"10 > 3: {10 > 3}")           # True
print(f"10 == 3: {10 == 3}")         # False
print(f"10 != 3: {10 != 3}")         # True
print(f"10 >= 10: {10 >= 10}")       # True

# 布尔运算
print(f"True and False: {True and False}")   # False
print(f"True or False: {True or False}")     # True
print(f"not True: {not True}")               # False


# ─────────────────────────────────────
# 5. 类型转换
# ─────────────────────────────────────
print("\n【5. 类型转换】")

# 字符串 → 数字
price_str = "99.8"
price = float(price_str)
print(f"字符串 '{price_str}' → 浮点数 {price}")

# 数字 → 字符串
num = 42
num_str = str(num)
print(f"整数 {num} → 字符串 '{num_str}'")

# 整数 → 浮点数
x = 10
y = float(x)
print(f"整数 {x} → 浮点数 {y}")


# ─────────────────────────────────────
# ✏️ 动手练习
# ─────────────────────────────────────
print("\n" + "=" * 50)
print("✏️ 动手练习")
print("=" * 50)

# 练习1：计算滑坡位移速率
# 已知：3天位移了 68.5mm，求平均每天位移多少？
total_displacement = 68.5  # mm
days = 3
rate = total_displacement / days
print(f"\n练习1: 平均位移速率 = {rate:.2f} mm/天")

# 练习2：字符串处理
sensor_name = "GNSS-01"
print(f"传感器编号: {sensor_name}")
print(f"类型: {sensor_name.split('-')[0]}")      # GNSS
print(f"编号: {sensor_name.split('-')[1]}")       # 01

# 练习3：你来试试
# TODO: 创建一个变量存储你的名字，用 f-string 打印 "你好，我是XXX！"
# TODO: 计算 2026 减去你的出生年份，打印 "我今年XX岁"

print("\n✅ 第一课完成！下一课：条件语句与循环")
