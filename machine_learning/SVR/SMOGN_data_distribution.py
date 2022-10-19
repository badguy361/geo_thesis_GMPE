import smogn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

TSMIP_df = pd.read_csv("../../../TSMIP_FF.csv")
TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn.csv")

TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
TSMIP_df = TSMIP_df[(TSMIP_df['Rrup'] < 1000) & (TSMIP_df['Rrup'] > 10)]
TSMIP_df = TSMIP_df[TSMIP_df['MW'] < 6.8]  # 試看看跟paper一樣的資料分布
TSMIP_df['lnRrup'] = np.log(TSMIP_df['Rrup'])
TSMIP_df['Mw_size'] = np.zeros(len(TSMIP_df['lnRrup']))
TSMIP_df['log10Vs30'] = np.log10(TSMIP_df['Vs30'])
TSMIP_df['log10Rrup'] = np.log10(TSMIP_df['Rrup'])
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RO", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RV", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NM", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NO", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("SS", "3")
TSMIP_df['fault.type'] = pd.to_numeric(TSMIP_df['fault.type'])
TSMIP_df['lnPGA(gal)'] = np.log(TSMIP_df['PGA'] * 980)

TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['Mw_size'] = np.zeros(len(TSMIP_smogn_df['lnRrup']))
TSMIP_smogn_df['log10Vs30'] = np.log10(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['log10Rrup'] = np.log10(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "RO", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "RV", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "NM", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "NO", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "SS", "3")
TSMIP_smogn_df['fault.type'] = pd.to_numeric(TSMIP_smogn_df['fault.type'])
TSMIP_smogn_df['lnPGA(gal)'] = np.log(TSMIP_smogn_df['PGA'] * 980)

#################################################################

TSMIP_smogn_ori_df = pd.read_csv("../../../TSMIP_smogn+oridata.csv")

TSMIP_smogn_ori_df['lnVs30'] = np.log(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['lnRrup'] = np.log(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['Mw_size'] = np.zeros(len(TSMIP_smogn_ori_df['lnRrup']))
TSMIP_smogn_ori_df['log10Vs30'] = np.log10(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['log10Rrup'] = np.log10(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("RO", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("RV", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("NM", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("NO", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("SS", "3")
TSMIP_smogn_ori_df['fault.type'] = pd.to_numeric(
    TSMIP_smogn_ori_df['fault.type'])
TSMIP_smogn_ori_df['lnPGA(gal)'] = np.log(TSMIP_smogn_ori_df['PGA'] * 980)


# 1. Rrup Mw distribution

# plt.grid(color='gray', linestyle = '--',linewidth=0.5)
# plt.scatter(TSMIP_df['Rrup'], TSMIP_df['MW'], marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
# plt.xscale("log")
# plt.xlim(1e-1,1e3)
# plt.xlabel('Rrup')
# plt.ylabel('Mw')
# plt.title('TSMIP_SMOGN Data distribution')
# plt.legend()
# plt.savefig(f'TSMIP_SMOGN Data distribution.png',dpi=300)
# plt.show()

# 2. Vs30 Number of ground motion distribution

# plt.grid(color='gray', linestyle = '--',linewidth=0.5)
# plt.hist(TSMIP_smogn_df['Vs30'], bins=20, edgecolor="yellow", color="green")
# plt.xlabel('Vs30(m/s)')
# plt.ylabel('Number of record')
# plt.ylim(0,9000)
# plt.title('TSMIP_SMOGN Vs30 distribution')
# plt.savefig('TSMIP_SMOGN Vs30 distribution.png',dpi=300)
# plt.show()