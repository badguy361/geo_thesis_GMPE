import sys
import numpy as np
import math
# append the path of the
# parent directory
sys.path.append("..")
from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
import pandas as pd
import matplotlib.pyplot as plt
import shap

target = "PGA"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
DSCon = pd.read_csv(f"../../../Distance Scaling Condition.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preprocess(TSMIP_df, target, True)
after_process_DSCon = model.preprocess(DSCon, target, False)

result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, 'lnVs30', 'MW',
                                   'lnRrup', 'fault.type', 'STA_Lon_X',
                                   'STA_Lat_Y')
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, 'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                 'STA_Lon_X', 'STA_Lat_Y')
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, 'lnVs30', 'MW', 'lnRrup',
                                    'fault.type', 'STA_Lon_X', 'STA_Lat_Y')
DSCon_data = model.split_dataset(after_process_DSCon, f'ln{target}(gal)',
                                 False, 'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                 'STA_Lon_X', 'STA_Lat_Y')

# result_ori[0](訓練資料)之shape : (29896,6) 為 29896筆 records 加上以下6個columns ['lnVs30','MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

originaldata_predicted_result = model.predicted_original(
    ML_model, original_data)

plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.predicted_distribution(result_ori[1], result_ori[3],
#                                        final_predict, fit_time, score)
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score)
# for i in range(len(DSCon['MW'])):
#     Mw = DSCon['MW'][i]
#     Vs30 = DSCon['Vs30'][i]
#     faulttype = DSCon['fault.type'][i]
#     plot_something.distance_scaling(i, ML_model, DSCon_data, Vs30, Mw, faulttype,
#                                     score)
seed = 555
plot_something.explainable(result_ori[1], ML_model, seed)

TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")
point = [119.5635611,21.90093889] # 最左下角測站為基準
STA_DIST = (((TSMIP_df["STA_Lat_Y"]-point[1])*110)**2 + ((TSMIP_df["STA_Lon_X"]-point[0])*101)**2)**(1/2)
TSMIP_df["STA_DIST"] = STA_DIST
TSMIP_df["STA_rank"] = TSMIP_df["STA_DIST"].rank(method='dense')

from collections import Counter
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
plt.title('STA ID')
plt.savefig("STA_ID.jpg",dpi=300)
plt.show()