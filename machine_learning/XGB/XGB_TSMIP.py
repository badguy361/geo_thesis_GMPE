import sys
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
# append the path of the
# parent directory
sys.path.append("..")
from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig

target = "Sa005"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
# DSCon = pd.read_csv(f"../../../Distance Scaling Condition.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preprocess(TSMIP_df, target, True)
# after_process_DSCon = model.preprocess(DSCon, "PGA", False)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)
# DSCon_data = model.split_dataset(after_process_DSCon, f'ln{"PGA"}(gal)', False,
#                                  *model_feture)

#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# score, feature_importances, fit_time, final_predict, ML_model = model.training(
#     target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2],
#     result_ori[3])

# originaldata_predicted_result = model.predicted_original(
#     ML_model, original_data)

PGA_model = pickle.load(open('model/XGB_Sa001.pkl', 'rb'))

plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.predicted_distribution(result_ori[1], result_ori[3],
#                                        final_predict, fit_time, score)
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score)

Mw = 6.35
Rrup = 30
Vs30 = 440
fault_type = 2
station_rank = 265
plot_something.distance_scaling(Vs30, Mw, Rrup, fault_type, station_rank,
                                original_data[0], original_data[1], PGA_model)
# seed = 18989
# plot_something.explainable(original_data[0], model_feture, ML_model, seed)

# model = [
#     'model/XGB_PGA.pkl', 'model/XGB_PGV.pkl', 'model/XGB_Sa001.pkl',
#     'model/XGB_Sa005.pkl', 'model/XGB_Sa01.pkl', 'model/XGB_Sa02.pkl',
#     'model/XGB_Sa05.pkl', 'model/XGB_Sa10.pkl', 'model/XGB_Sa30.pkl',
#     'model/XGB_Sa40.pkl', 'model/XGB_Sa100.pkl'
# ]
# plot_something.respond_spetrum(Vs30, Mw, Rrup, fault_type, station_rank,
#                                *model)
