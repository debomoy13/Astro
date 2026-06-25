#this is a one time runner code , repeated running will cause repeated downloads
import os
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve
import numpy as np

koi = pd.read_csv("modified datasets/koi_cumulative_labeled.csv" , comment='#')

subset=(
    koi.groupby("signal_class", group_keys=False).head(10)
)
os.makedirs("dataset", exist_ok=True)

records=[]

for _,row in subset.iterrows():
    kepid = row["kepid"]
    label = row["signal_class"]
    lc = search_lightcurve(f"KIC {kepid}", mission="kepler").download()
    lc = lc.remove_nans()
    lc = lc.normalize()

    np.savez(
        f"dataset/{kepid}.npz",
        time=lc.time.value,
        flux=lc.flux.value
        )

    records.append({
        "kepid": kepid,
        "label": label,
        "file": f"dataset/{kepid}.npz"
        })

    print(f"Downloaded {kepid}")

pd.DataFrame(records).to_csv("dataset_index.csv", index=False)