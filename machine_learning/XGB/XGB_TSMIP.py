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

target = "PGV"

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
original_EQID_data = model.split_dataset(after_process_ori_data,
                                         f'ln{target}(gal)', False, 'EQ_ID',
                                         'lnVs30', 'MW', 'lnRrup',
                                         'fault.type', 'STA_Lon_X',
                                         'STA_Lat_Y')
# result_ori[0](訓練資料)之shape : (29896,6) 為 29896筆 records 加上以下6個columns ['lnVs30','MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

originaldata_predicted_result = model.predicted_original(
    ML_model, original_data)

residual = final_predict - result_ori[3]

plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.train_test_distribution(result_ori[1], result_ori[3],
#                                        final_predict, fit_time, score)
plot_something.residual(result_ori[1], result_ori[3], final_predict, score) 
plot_something.measured_predict(result_ori[3], final_predict, score) 
# minVs30 = 480
# maxVs30 = 760
# minMw = 4.0
# maxMw = 5.0
# faulttype = 1
# plot_something.distance_scaling(original_data, originaldata_predicted_result, # 應該要改成只有test的資料
#                                 minVs30, maxVs30, minMw, maxMw, faulttype,
#                                 score,5)

final_predict_df = pd.DataFrame(final_predict, columns=['predicted'])
ans_predict_df = pd.DataFrame(result_ori[3], columns=['ans'])
features_predict_df = pd.DataFrame(
    result_ori[1],
    columns=['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y'])
features_predict_df = pd.DataFrame(
    original_EQID_data[1],
    columns=['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y'])

# 重複數據 273顆地震
# summeries = {'PGV': 'mean'}
# TSMIP_df.groupby(by="EQ_ID").agg(summeries).reset_index()