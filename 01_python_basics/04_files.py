"""
第四课：文件读写与异常处理
=====================
把数据存到文件里，读取已有文件，处理出错情况

运行方式: python 04_files.py
"""

import random
import os

print("=" * 50)
print("第四课：文件读写与异常处理")
print("=" * 50)

# ─────────────────────────────────────
# 1. 写入文本文件
# ─────────────────────────────────────
print("\n【1. 写入文件】")

# 准备一些模拟监测数据
random.seed(42)
records = []
for day in range(1, 11):
    records.append(f"第{day:2d}天, GNSS-01, 位移{random.uniform(2, 20):.2f}mm, 正常\n")

# 写入文件（自动创建文件）
with open("monitor_data.txt", "w", encoding="utf-8") as f:
    f.write("日期, 传感器, 位移, 状态\n")
    f.write("-" * 40 + "\n")
    for r in records:
        f.write(r)

print("✅ 数据已写入 monitor_data.txt")
print("   内容:")
with open("monitor_data.txt", "r", encoding="utf-8") as f:
    print(f.read())


# ─────────────────────────────────────
# 2. 读取文件的不同方式
# ─────────────────────────────────────
print("【2. 读取文件】")

# 方式1：读取全部内容
with open("monitor_data.txt", "r", encoding="utf-8") as f:
    content = f.read()
print(f"文件总字符数: {len(content)}")

# 方式2：逐行读取（推荐，内存友好）
print("\n逐行读取并解析:")
parsed_data = []
with open("monitor_data.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去掉首尾空白
        if line and not line.startswith("日期") and not line.startswith("-"):
            # 解析每行数据
            parts = line.split(", ")
            parsed_data.append({
                "day": int(parts[0].replace("第", "").replace("天", "")),
                "sensor": parts[1],
                "displacement": float(parts[2].replace("位移", "").replace("mm", "")),
                "status": parts[3]
            })

print(f"成功解析 {len(parsed_data)} 条记录")
print(f"示例: {parsed_data[0]}")


# ─────────────────────────────────────
# 3. 写入 CSV 文件（结构化数据）
# ─────────────────────────────────────
print("\n【3. CSV 文件操作】")

import csv

# 生成更完整的数据
random.seed(42)
headers = ["日期", "传感器", "位移(mm)", "降雨量(mm)", "库水位(m)", "预警等级"]
rows = []
for day in range(1, 31):
    disp = round(random.uniform(1, 25), 2)
    rain = round(random.uniform(0, 120), 1)
    water = round(random.uniform(155, 178), 1)
    level = "危险" if disp > 15 else ("预警" if disp > 8 else "正常")
    rows.append([f"2026-03-{day:02d}", f"GNSS-{(day-1)%4+1:02d}", disp, rain, water, level])

# 写入 CSV
with open("monitor_data.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print("✅ 已写入 monitor_data.csv")

# 读取 CSV
with open("monitor_data.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    csv_data = list(reader)

print(f"共读取 {len(csv_data)} 条记录")
print(f"示例: {csv_data[0]}")

# 统计各等级天数
from collections import Counter
levels = Counter(row["预警等级"] for row in csv_data)
print(f"预警统计: {dict(levels)}")


# ─────────────────────────────────────
# 4. JSON 文件（程序间交换数据）
# ─────────────────────────────────────
print("\n【4. JSON 文件操作】")

import json

# 创建传感器配置
config = {
    "project": "黄土坡滑坡监测",
    "sensors": [
        {"id": "GNSS-01", "lat": 30.92, "lon": 110.83, "active": True},
        {"id": "GNSS-02", "lat": 30.93, "lon": 110.84, "active": True},
        {"id": "GNSS-03", "lat": 30.91, "lon": 110.82, "active": False},
    ],
    "thresholds": {
        "displacement_normal": 8,
        "displacement_warning": 15,
        "displacement_danger": 25,
    },
    "update_interval": "1h"
}

# 写入 JSON
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("✅ 配置已写入 config.json")
print(json.dumps(config, ensure_ascii=False, indent=2))

# 读取 JSON
with open("config.json", "r", encoding="utf-8") as f:
    loaded_config = json.load(f)

print(f"\n项目名: {loaded_config['project']}")
print(f"传感器数量: {len(loaded_config['sensors'])}")


# ─────────────────────────────────────
# 5. 异常处理 try-except
# ─────────────────────────────────────
print("\n【5. 异常处理】")

# 常见错误类型
errors_to_try = [
    ("除零错误", lambda: 10 / 0),
    ("索引越界", lambda: [1, 2, 3][10]),
    ("类型错误", lambda: "hello" + 5),
    ("文件不存在", lambda: open("不存在的文件.txt", "r")),
]

for name, func in errors_to_try:
    try:
        func()
    except Exception as e:
        print(f"  ❌ {name}: {type(e).__name__} - {e}")


# 实际应用：安全的文件读取
def safe_read_file(filepath):
    """安全读取文件，处理各种异常"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 '{filepath}' 不存在"
    except PermissionError:
        return f"错误：没有权限读取 '{filepath}'"
    except Exception as e:
        return f"错误：{type(e).__name__} - {e}"

print("\n安全读取测试:")
print(f"  {safe_read_file('monitor_data.txt')[:30]}...")
print(f"  {safe_read_file('not_exist.txt')}")


# ─────────────────────────────────────
# 6. 综合实战：监测数据报告生成器
# ─────────────────────────────────────
print("\n【6. 实战：生成监测报告】")

def generate_report(data, filename="report.txt"):
    """根据监测数据生成文本报告"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("         滑坡监测日报\n")
        f.write("=" * 50 + "\n\n")

        # 基本信息
        total = len(data)
        f.write(f"监测天数: {total}天\n")

        # 位移统计
        disps = [float(row["位移(mm)"]) for row in data]
        f.write(f"累计位移: {sum(disps):.2f}mm\n")
        f.write(f"平均位移: {sum(disps)/len(disps):.2f}mm/天\n")
        f.write(f"最大位移: {max(disps):.2f}mm\n")
        f.write(f"最小位移: {min(disps):.2f}mm\n\n")

        # 预警统计
        f.write("预警统计:\n")
        for level, count in levels.items():
            bar = "█" * count
            f.write(f"  {level}: {bar} {count}天\n")

        # 危险日详情
        f.write("\n危险日详情:\n")
        for row in data:
            if row["预警等级"] == "危险":
                f.write(f"  {row['日期']} | {row['传感器']} | "
                        f"位移 {row['位移(mm)']}mm\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("报告生成时间: 2026-04-03\n")

    return filename

report_file = generate_report(csv_data)
print(f"✅ 报告已生成: {report_file}")

# 显示报告内容
with open(report_file, "r", encoding="utf-8") as f:
    print("\n" + f.read())


# ─────────────────────────────────────
# 清理临时文件（可选）
# ─────────────────────────────────────
print("【清理临时文件】")
for f in ["monitor_data.txt", "monitor_data.csv", "config.json", "report.txt"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  已删除: {f}")


# ─────────────────────────────────────
# ✏️ 动手练习
# ─────────────────────────────────────
print("\n" + "=" * 50)
print("✏️ 动手练习")
print("=" * 50)

# TODO: 写一个函数，把一个列表保存到文件，再读回来
# TODO: 用 try-except 处理用户输入：让用户输入一个数字，如果不是数字就提示重新输入

print("""
Python 基础四课全部完成！📚

你已掌握:
  ✅ 变量与数据类型（字符串、数字、布尔）
  ✅ 条件判断与循环（if、for、while）
  ✅ 函数与数据结构（def、list、dict）
  ✅ 文件读写与异常处理（txt、csv、json、try-except）

下一步: 02_data_processing/ 学习 NumPy 和 Pandas！
""")
