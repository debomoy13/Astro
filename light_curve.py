import os
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve

koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')

subset=(
    koi.groupby("signal_class", group_keys=False).head(100)
)
os.makedirs("dataset", exist_ok=True)

records=[]

for row in subset.itterows():
    kepid = row["kepid"]
    label = row["signal_class"]
    lc = search_lightcurve(f"KIC {kepid}", mission="kepler").download()
