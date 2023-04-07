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
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target)
after_process_ori_data = model.preprocess(TSMIP_df, target)
after_process_DSCon = model.preprocess(DSCon, target)

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

df = pd.DataFrame(
    result_ori[1],
    columns=['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_Lon_X', 'STA_Lat_Y'])
explainer = shap.Explainer(ML_model)
shap_values = explainer(df)
shap.summary_plot(shap_values, df, show=False)
plt.savefig("summary_plot.jpg", bbox_inches='tight', dpi=300)

# waterfall
shap.plots.waterfall(shap_values[0], show=False)  # 單筆資料解釋:第1筆資料解釋
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.savefig("shap_waterfall.jpg", bbox_inches='tight', dpi=300)
# force plot
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values.values[0, :],
                df.iloc[0, :], show=False,matplotlib=True)
plt.savefig("force_plot.jpg", bbox_inches='tight', dpi=300)

# bar plot
fig = plt.figure()
shap.plots.bar(shap_values, show=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.savefig("shap_bar.jpg", bbox_inches='tight', dpi=300)
# scatter plot
shap.plots.scatter(shap_values[:, "MW"],
                   color=shap_values[:, "lnRrup"],
                   show=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.savefig("shap_scatter.jpg", bbox_inches='tight', dpi=300)

shap.plots.scatter(shap_values[:, "lnRrup"],
                   color=shap_values[:, "MW"],
                   show=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.savefig("shap_scatte.jpg", bbox_inches='tight', dpi=300)