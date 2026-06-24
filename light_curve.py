import os
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve

koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for kepid in koi["kepid"]:

    lc= search_lightcurve(
        f"KIC {sample("kepid")}",
        mission ="Kepler"
        ).download()
    lc.plot()
    plt.savefig(os.path.join(output_dir, f"KIC_{kepid}.png"))
    plt.close()

