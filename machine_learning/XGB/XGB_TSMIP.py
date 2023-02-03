import sys
import numpy as np
# append the path of the
# parent directory
sys.path.append("..")

from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
import pandas as pd
import matplotlib.pyplot as plt

target = "PGV"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target)
after_process_ori_data = model.preprocess(TSMIP_df, target)
result_SMOGN = model.split_dataset(TSMIP_smogn_df, f'ln{target}(gal)', True,
                                   'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                   'STA_Lon_X', 'STA_Lat_Y')
result_ori = model.split_dataset(TSMIP_df, f'ln{target}(gal)', True, 'lnVs30', 'MW',
                                 'lnRrup', 'fault.type', 'STA_Lon_X',
                                 'STA_Lat_Y')
original_data = model.split_dataset(TSMIP_df, f'ln{target}(gal)', False, 'lnVs30',
                                    'MW', 'lnRrup', 'fault.type', 'STA_Lon_X',
                                    'STA_Lat_Y')
# result_ori[0](訓練資料)之shape : (29896,6) 為 29896筆 records 加上以下6個columns ['lnVs30','MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

originaldata_predicted_result = model.predicted_original(
    ML_model, original_data)

plot_something = plot_fig("XGBooster", "XGB", "SMOGN",target)
# plot_something.train_test_distribution(result_ori[1], result_ori[3], final_predict, fit_time, score)
# plot_something.residual(result_ori[1], result_ori[3], final_predict, score)
# plot_something.measured_predict(result_ori[3], final_predict, score)
minVs30 = 480
maxVs30 = 760
minMw = 7.0
maxMw = 8.0
faulttype = 1
plot_something.distance_scaling(original_data, originaldata_predicted_result,
                                minVs30, maxVs30, minMw, maxMw, faulttype,
                                score)
