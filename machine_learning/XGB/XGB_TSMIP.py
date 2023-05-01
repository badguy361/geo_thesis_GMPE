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

target = "Sa05"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
DSCon = pd.read_csv(f"../../../Distance Scaling Condition.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preprocess(TSMIP_df, target, True)
after_process_DSCon = model.preprocess(DSCon, "PGA", False)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)
DSCon_data = model.split_dataset(after_process_DSCon, f'ln{"PGA"}(gal)', False,
                                 *model_feture)

#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2],
    result_ori[3])

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
seed = 16044
plot_something.explainable(original_data[0], model_feture, ML_model, seed)

# 加载模型
PGA_model = pickle.load(open("XGB_PGA.pkl", 'rb'))
PGV_model = pickle.load(open("XGB_PGV.pkl", 'rb'))
Sa001_model = pickle.load(open("XGB_Sa001.pkl", 'rb'))
Sa005_model = pickle.load(open("XGB_Sa005.pkl", 'rb'))
Sa01_model = pickle.load(open("XGB_Sa01.pkl", 'rb'))
Sa02_model = pickle.load(open("XGB_Sa02.pkl", 'rb'))
Sa05_model = pickle.load(open("XGB_Sa05.pkl", 'rb'))
Sa10_model = pickle.load(open("XGB_Sa10.pkl", 'rb'))
Sa30_model = pickle.load(open("XGB_Sa30.pkl", 'rb'))
Sa40_model = pickle.load(open("XGB_Sa40.pkl", 'rb'))
Sa100_model = pickle.load(open("XGB_Sa100.pkl", 'rb'))

# 使用模型进行预测
PGA_predict = np.exp(PGA_model.predict(DSCon_data[0])) / 980
PGV_predict = np.exp(PGV_model.predict(DSCon_data[0])) / 980
Sa001_predict = np.exp(Sa001_model.predict(DSCon_data[0])) / 980
Sa005_predict = np.exp(Sa005_model.predict(DSCon_data[0])) / 980
Sa01_predict = np.exp(Sa01_model.predict(DSCon_data[0])) / 980
Sa02_predict = np.exp(Sa02_model.predict(DSCon_data[0])) / 980
Sa05_predict = np.exp(Sa05_model.predict(DSCon_data[0])) / 980
Sa10_predict = np.exp(Sa10_model.predict(DSCon_data[0])) / 980
Sa30_predict = np.exp(Sa30_model.predict(DSCon_data[0])) / 980
Sa40_predict = np.exp(Sa40_model.predict(DSCon_data[0])) / 980
Sa100_predict = np.exp(Sa100_model.predict(DSCon_data[0])) / 980

fault_type = ["REV", "NM", "SS"]
plt.grid(which="both", axis="both", linestyle="-", linewidth=0.5, alpha=0.5)
plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
    Sa001_predict[0], Sa005_predict[0], Sa01_predict[0], Sa02_predict[0],
    Sa05_predict[0], Sa10_predict[0], Sa30_predict[0], Sa40_predict[0],
    Sa100_predict[0]
],label=fault_type[0])

# for i in range(0, 3):
#     plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
#         Sa001_predict[i], Sa005_predict[i], Sa01_predict[i], Sa02_predict[i],
#         Sa05_predict[i], Sa10_predict[i], Sa30_predict[i], Sa40_predict[i],
#         Sa100_predict[i]
#     ],label=fault_type[i])

plt.title(
    f"M = {DSCon['MW'][0]}, Rrup = {DSCon['Rrup'][0]}km, Vs30 = {DSCon['Vs30'][0]}m/s"
)
plt.xlabel("Period(s)")
plt.ylabel("PSA(g)")
plt.ylim(10e-6, 1)
plt.xlim(0.01, 10.0)
plt.yscale("log")
plt.xscale("log")
plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
           [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
plt.legend()
plt.savefig(
    f"response spectra-{DSCon['MW'][0]} {DSCon['Rrup'][0]} {DSCon['Vs30'][0]} {DSCon['fault.type'][0]}.jpg",
    dpi=300)
