import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import sys
import xgboost as xgb
# append the path of the
# parent directory
sys.path.append("..")
from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig

#? parameters
target = "PGA"
Mw = 7
Rrup = 30
Vs30 = 360
fault_type = 90
# station_rank = 265
# model_name = [
#     'model/XGB_PGA.pkl', 'model/XGB_PGV.pkl', 'model/XGB_Sa001.pkl',
#     'model/XGB_Sa005.pkl', 'model/XGB_Sa01.pkl', 'model/XGB_Sa02.pkl',
#     'model/XGB_Sa03.pkl', 'model/XGB_Sa05.pkl', 'model/XGB_Sa10.pkl',
#     'model/XGB_Sa30.pkl', 'model/XGB_Sa40.pkl', 'model/XGB_Sa100.pkl'
# ]
# seed = 18989

score = 0.88 # note

#? data preprocess
TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preprocess(TSMIP_df, target, True)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)

#? model train
#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# score, feature_importances, fit_time, final_predict, ML_model = model.training(
#     target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2],
#     result_ori[3])

#? model predicted
booster = xgb.Booster()
booster.load_model(f'XGB_{target}.json')
# booster.predict(xgb.DMatrix([(np.log(760), 7, np.log(200), -45, 256)]))
originaldata_predicted_result = model.predicted_original(
    booster, original_data)

#? plot figure
# plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# # plot_something.predicted_distribution(result_ori[1], result_ori[3],
# #                                        final_predict, fit_time, score)
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score)
# plot_something.distance_scaling(Vs30, Mw, Rrup, fault_type, station_rank,
#                                 original_data[0], original_data[1], ML_model)
# plot_something.respond_spetrum(Vs30, Mw, Rrup, fault_type, station_rank,
#                                False
#                                , *model_name)
# plot_something.explainable(original_data[0], model_feture, ML_model, seed)
