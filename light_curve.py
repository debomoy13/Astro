import os
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve

koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')

subset=(
    koi.groupby("signal_class", group_keys=False).head(100)
)
os.makedirs("dataset", exit_ok=True)

records=[]

