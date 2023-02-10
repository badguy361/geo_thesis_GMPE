import sys
import numpy as np
# append the path of the
# parent directory
sys.path.append("..")
from sklearn.model_selection import train_test_split
from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt

target = "PGA"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target)
after_process_ori_data = model.preprocess(TSMIP_df, target)
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
# result_ori[0](訓練資料)之shape : (29896,6) 為 29896筆 records 加上以下6個columns ['lnVs30','MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

originaldata_predicted_result = model.predicted_original(
    ML_model, original_data)

residual = final_predict - result_ori[3]

# plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.train_test_distribution(result_ori[1], result_ori[3],
#                                        final_predict, fit_time, score)
# plot_something.residual(result_ori[1], result_ori[3], final_predict, score) 
# plot_something.measured_predict(result_ori[3], final_predict, score) # 應該要改成全部的資料
# minVs30 = 480
# maxVs30 = 760
# minMw = 4.0
# maxMw = 5.0
# faulttype = 1
# plot_something.distance_scaling(original_data, originaldata_predicted_result, 
#                                 minVs30, maxVs30, minMw, maxMw, faulttype,
#                                 score,5)

originaldata_predicted_result_df = pd.DataFrame(originaldata_predicted_result, columns=['predicted'])
total_data_df = pd.concat([after_process_ori_data,originaldata_predicted_result_df],axis=1)
total_data_df["residual"] = np.abs((np.exp(total_data_df["predicted"]) - np.exp(total_data_df["lnPGA(gal)"]))/980)

# 重複數據 273顆地震
summeries = {'residual': 'mean','MW': 'max'}
inter_event = total_data_df.groupby(by="EQ_ID").agg(summeries).reset_index()
inter_event = inter_event.rename(columns = {'residual':'inter_event_residual','MW':'Mw'})

plt.grid(linestyle=':', color='darkgrey')
plt.scatter(inter_event['Mw'],
            inter_event['inter_event_residual'],
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.show()

total_data_df = pd.merge(total_data_df, inter_event, how='left', on=['EQ_ID'])
total_data_df['intra_event_residual'] = total_data_df['residual'] - total_data_df['inter_event_residual']

xticks = [0, 50, 100, 150, 200, 250, 300, 400, 500]
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(total_data_df['Rrup'],
            total_data_df['intra_event_residual'],
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.xticks(xticks)
plt.show()
