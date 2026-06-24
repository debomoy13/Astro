import pandas as pd
koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')
sample=koi.iloc[0]
print(sample["kepid"])