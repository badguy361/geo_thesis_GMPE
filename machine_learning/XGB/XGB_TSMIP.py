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

c1 = -1.4526
c2 = 1.06
c3 = 1.4379
cn = 12.1487
cm = 5.50455
c1a = 0.1379
c1c = 0.04273
c1b = 0
c1d = -0.1653
c4 = -2.1
c4a = 0.5
cRB = 50
dp = -6.7852
c5 = 6.4551
cHM = 3.0956
c6 = 0.4908
c7 = 0.0080
c7b = 0.0210
c8 = 0
c8a = 0.2695
c8b = 0.4833
c9 = 0.9228
c9a = 0.1202
c9b = 6.8607
c11 = -0.108
c11b = 0.196
cg1 = -0.0088
cg2 = -0.0071
cg3 = 4.2256
o1 = -0.5107
o2 = -0.1417
o3 = -0.007
o4 = 0.1022
o5 = 0.0744
o6 = 300
tau = 0.3730
oss = 0.4397
os2s = 0.3149
ln_y = c1 + (c1a + c1c / np.cosh(2 * np.max([M - 4.5, 0]))) * F_RV + (
    c1b + c1d / np.cosh(2 * np.max([M - 4.5, 0]))
) * F_NM + (c7 + c7b / np.cosh(2 * np.max([M - 4.5, 0]))) * Z_tor + (
    c11 + c11b / np.cosh(2 * np.max([M - 4.5, 0]))
) * np.cos(delta) + c2 * (M - 6) + ((c2 - c3) / cn) * (
    (np.log(1 + np.exp(cn * (cm - M))))**
    2) + c8 * np.max([1 - np.max([Rrup - 40, 0]) / 30, 0]) * np.min(
        [np.max([M - 5.55, 0]) / 0.8, 1]) * np.exp(
            -c8a * (Mi - c8b) * deltaDPP) + c9 * F_HW * np.cos(delta) * (
                c9a + (1 - c9a) * np.tanh(Rx / c9b)) * (1 - (
                    (Rjb**2 + Ztor**2)**(1 / 2)) / (Rrup + 1)) + c4 * np.log(
                        Rrup + c5 * np.cosh(c6 * np.max([M - cHM, 0]))
                    ) + (c4a - c4) * np.log((Rrup**2 + cRB**2)**(1 / 2)) + (
                        cg1 + cg2 / np.cosh(np.max([Mi - cg3, 0]))
                    ) * Rrup + o1 * np.min([np.log(Vs30 / 1130), 0]) + o2 * (
                        np.exp(o3 * np.min([Vs30, 1130]) - 360) -
                        np.exp(o3 * (1130 - 360))) * np.log(
                            (y1130 * np.exp(ie) + o4) /
                            o4) + o5 * (1 - np.exp(-deltaZ_10 / o6)) + ie + ia
