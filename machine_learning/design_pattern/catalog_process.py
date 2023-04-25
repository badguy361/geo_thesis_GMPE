import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")
point = [119.5635611,21.90093889] # 最左下角為基準
STA_DIST = (((TSMIP_df["STA_Lat_Y"]-point[1])*110)**2 + ((TSMIP_df["STA_Lon_X"]-point[0])*101)**2)**(1/2)
TSMIP_df["STA_DIST"] = STA_DIST
TSMIP_df["STA_rank"] = TSMIP_df["STA_DIST"].rank(method='dense')

for i in range(1,38):
    plt.figure(figsize=(20, 6))
    plt.bar(list(dict(Counter(TSMIP_df["STA_ID"])).keys())[20*(i-1):20*i],list(dict(Counter(TSMIP_df["STA_ID"])).values())[20*(i-1):20*i])
    plt.title(f'STA Distribution ID_{20*(i-1)}-{20*i}')
    plt.savefig(f"STA_ID_distribution_{20*(i-1)}-{20*i}.jpg",dpi=300)
# len(TSMIP_df["STA_ID"].unique())
plt.scatter(TSMIP_df["STA_Lon_X"],
            TSMIP_df["STA_Lat_Y"],
            c=TSMIP_df["STA_rank"],
                    cmap='bwr')
plt.colorbar(extend='both', label='number value')
plt.plot(point[0],point[1],"*",color='black')
plt.title('TSMIP Station located')
plt.savefig("../XGB/STA_ID.jpg",dpi=300)
plt.show()