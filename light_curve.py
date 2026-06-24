import pandas as pd
import matplotlib.pyplot as plt
koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')
sample=koi.iloc[0]
print(sample["kepid"])

from lightkurve import search_lightcurve

lc= search_lightcurve(
    "KIC 10797460",
    mission ="Kepler"
).download()
lc.plot()
plt.show()

