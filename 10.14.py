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
=COUNTIFS(MIN(D2:D883,G2:G883,J2:J883),"<=0")
=SUMPRODUCT(--(MIN(D2:D883, G2:G883, J2:J883) <= 0))
=SUMPRODUCT(--((D2:D883<=0) + (G2:G883<=0) + (J2:J883<=0) > 0))
=SUMPRODUCT((D1:D1000<0) + (G1:G1000<0) + (J1:J1000<0)) / (ROWS(D1:D1000) * 3)

10.16：
=MIN(D:D, G:G, J:J)
=AVERAGEIF((D:D, G:G, J:J), "<0")
=AVERAGE(IF((D:D<0) + (G:G<0) + (J:J<0), (D:D, G:G, J:J)))
=MIN(D:D, G:G, J:J)
=AVERAGE(IF((D:D<0) + (G:G<0) + (J:J<0), IF((D:D<0), D:D, IF((G:G<0), G:G, IF((J:J<0), J:J)))))
=SUMPRODUCT( ((D2:D883<0) + (G2:G883<0) + (J2:J883<0) >= 3) * ((D2:D883<=-0.02) + (G2:G883<=-0.02) + (J2:J883<=--0.02) >=0)) / SUMPRODUCT( ((D2:D883<0) + (G2:G883<0) + (J2:J883<0) >= 2) * ((D2:D883<=-0.02) + (G2:G883<=-0.02) + (J2:J883<=-0.02) >=0))
=SUMPRODUCT( ((D2:D883<0) + (G2:G883<0) + (J2:J883<0) >= 3) / 2224)
