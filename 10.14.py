import pandas as pd

# 读取Excel文件
df = pd.read_excel('your_file.xlsx')

# 筛选出“好”的行
good_rows = df.apply(lambda row: any(row['C':'O'] > row['B']), axis=1)

# 计算“好”行的比例
good_ratio = good_rows.sum() / len(df)

print(f"好行的比例为: {good_ratio:.2%}")
=IF(MAX(C2:O2) > B2, 1, 0)
=SUM(P2:P2225) / (COUNTA(B2:B2225))
=MAX(C2:O2) - B2
=AVERAGEIF(R2:R2225, ">0")
