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

target = "PGA"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
DSCon = pd.read_csv(f"../../../Distance Scaling Condition.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preprocess(TSMIP_df, target, True)
after_process_DSCon = model.preprocess(DSCon, target, False)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)
DSCon_data = model.split_dataset(after_process_DSCon, f'ln{target}(gal)',
                                 False, *model_feture)

#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2],
    result_ori[3])

# originaldata_predicted_result = model.predicted_original(
#     ML_model, original_data)

# plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
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
# seed = 555
# plot_something.explainable(result_ori[1], model_feture, ML_model, seed)

# 加载模型
PGA_model = pickle.load(open("XGB_PGA.pkl", 'rb'))
PGV_model = pickle.load(open("XGB_PGV.pkl", 'rb'))
Sa10_model = pickle.load(open("XGB_Sa10.pkl", 'rb'))
Sa40_model = pickle.load(open("XGB_Sa40.pkl", 'rb'))
Sa02_model = pickle.load(open("XGB_Sa02.pkl", 'rb'))

# 使用模型进行预测
PGA_predict = PGA_model.predict(DSCon_data[0])
PGV_predict = PGV_model.predict(DSCon_data[0])
Sa10_predict = Sa10_model.predict(DSCon_data[0])
Sa40_predict = Sa40_model.predict(DSCon_data[0])
Sa02_predict = Sa02_model.predict(DSCon_data[0])

plt.plot([1, 2, 3, 4, 5], [
    PGA_predict[0], PGV_predict[0], Sa02_predict[0], Sa10_predict[0],
    Sa40_predict[0]
])
