import pandas as pd

# 读取Excel文件
df = pd.read_excel('your_file.xlsx')

# 筛选出“好”的行
good_rows = df.apply(lambda row: any(row['C':'O'] > row['B']), axis=1)

# 计算“好”行的比例
good_ratio = good_rows.sum() / len(df)

print(f"好行的比例为: {good_ratio:.2%}")
