import smogn
import pandas as pd
import numpy as np
from process_train import dataprocess
import matplotlib.pyplot as plt

"""
Compare synthesized data and original data
""" 
# target = "PGA"
# TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
# TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
# model = dataprocess()
# after_process_ori_data = model.preProcess(TSMIP_df, target, True)
# after_process_SMOGN_data = model.preProcess(TSMIP_smogn_df, target, False)
# model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# model_feture_old = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# result_SMOGN = model.splitDataset(after_process_SMOGN_data,
#                                    f'ln{target}(gal)', False, *model_feture_old)
# original_data = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
#                                     False, *model_feture)

# # original box plot
# o0 = np.swapaxes(original_data[0],0,1)[0]
# o1 = np.swapaxes(original_data[0],0,1)[1]
# o2 = np.swapaxes(original_data[0],0,1)[2]
# o3 = np.swapaxes(original_data[0],0,1)[3]
# o4 = np.swapaxes(original_data[0],0,1)[4]
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(15, 5))
# ax1.boxplot(o0)
# ax1.set_xticklabels(['lnVs30'])
# ax2.boxplot(o1)
# ax2.set_xticklabels(['MW'])
# ax3.boxplot(o2)
# ax3.set_xticklabels(['lnRrup'])
# ax4.boxplot(o3)
# ax4.set_xticklabels(['fault.type'])
# ax4.set_yticks([90,-90,0])
# ax4.set_yticklabels(["RE","NO","SS"])
# ax5.boxplot(o4)
# ax5.set_xticklabels(['STA_rank'])
# plt.suptitle('Boxplots about TSMIP data')
# plt.savefig('TSMIP boxplots.png',dpi=300)

# # SMOGN box plot
# o0 = np.swapaxes(result_SMOGN[0],0,1)[0]
# o1 = np.swapaxes(result_SMOGN[0],0,1)[1]
# o2 = np.swapaxes(result_SMOGN[0],0,1)[2]
# o3 = np.swapaxes(result_SMOGN[0],0,1)[3]
# o4 = np.swapaxes(result_SMOGN[0],0,1)[4]
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=(15, 5))
# ax1.boxplot(o0)
# ax1.set_xticklabels(['lnVs30'])
# ax2.boxplot(o1)
# ax2.set_xticklabels(['MW'])
# ax3.boxplot(o2)
# ax3.set_xticklabels(['lnRrup'])
# ax4.boxplot(o3)
# ax4.set_xticklabels(['fault.type'])
# ax4.set_yticks([90,-90,0])
# ax4.set_yticklabels(["RE","NO","SS"])
# ax5.boxplot(o4)
# ax5.set_xticklabels(['STA_rank'])
# plt.suptitle('Boxplots about TSMIP-SMOGN data')
# plt.savefig('TSMIP-SMOGN boxplots.png',dpi=300)

"""
Synthesized data
""" 
# target = "Sa001"
for target in ["Sa03"]:
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_pre_smogn_{target}.csv")

    ## conduct smogn ##這步大概要兩個小時
    TSMIP_smogn = smogn.smoter(
        data = TSMIP_df,
        y = target
    )
    TSMIP_smogn.to_csv(f"../../../TSMIP_smogn_{target}.csv",index=False)
