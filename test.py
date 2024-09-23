import pandas as pd

# 读取Excel文件
file_path = 'your_file.xlsx'  # 请替换为您的文件路径
df = pd.read_excel(file_path)

# 初始化计数器
count_negative_4 = 0
count_negative_7 = 0
count_negative_10 = 0
count_all_negative = 0

# 遍历第2行到第883行的数据
for index, row in df.iterrows():
    if index >= 1 and index <= 882:
        # 检查第4列、第7列、第10列的值是否小于0
        if row[3] < 0:
            count_negative_4 += 1
        if row[6] < 0:
            count_negative_7 += 1
        if row[9] < 0:
            count_negative_10 += 1
        # 检查这三列的值是否都小于0
        if row[3] < 0 and row[6] < 0 and row[9] < 0:
            count_all_negative += 1

# 输出结果
print(f"第4列中元素值小于0的个数: {count_negative_4}")
print(f"第7列中元素值小于0的个数: {count_negative_7}")
print(f"第10列中元素值小于0的个数: {count_negative_10}")
print(f"存在某一行，该行这三列的值都小于0的行数: {count_all_negative}")
