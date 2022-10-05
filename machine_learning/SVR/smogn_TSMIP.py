import smogn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygmt

## load data
# TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")

## conduct smogn ##這步大概要兩個小時
# TSMIP_smogn = smogn.smoter(
#     data = TSMIP_df, 
#     y = "PGA"
# )
# TSMIP_smogn.to_csv("../../../TSMIP_smogn.csv",index=False)

TSMIP_df = pd.read_csv("../../../TSMIP_smogn.csv")


TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
TSMIP_df['lnRrup'] = np.log(TSMIP_df['Rrup'])
TSMIP_df['Mw_size'] = np.zeros(len(TSMIP_df['lnRrup']))
################### Rrup Mw distribution ###########################

plt.grid(color='gray', linestyle = '--',linewidth=0.5)
plt.scatter(TSMIP_df['Rrup'], TSMIP_df['MW'], marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.xscale("log")
plt.xlim(1e-1,1e3)
plt.xlabel('Rrup')
plt.ylabel('Mw')
plt.title('TSMIP_SMOGN Data distribution')
plt.legend()
plt.savefig(f'TSMIP_SMOGN Data distribution.png',dpi=300)
plt.show()

################### Vs30 Number of ground motion ###########################

plt.grid(color='gray', linestyle = '--',linewidth=0.5)
plt.hist(TSMIP_df['Vs30'], bins=20, edgecolor="yellow", color="green")
plt.xlabel('Vs30(m/s)')
plt.ylabel('Number of record')
plt.title('TSMIP_SMOGN Vs30 distribution')
plt.legend()
plt.savefig(f'TSMIP_SMOGN Vs30 distribution.png',dpi=300)
plt.show()

