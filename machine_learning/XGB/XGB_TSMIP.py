import sys
import numpy as np
import math
# append the path of the
# parent directory
sys.path.append("..")
from sklearn.model_selection import train_test_split
from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import pandas as pd
import matplotlib.pyplot as plt

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
plot_something.residual(original_data[0], original_data[1],
                        originaldata_predicted_result, after_process_ori_data,
                        score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score)

# Mw = DSCon['MW'][0]
# Vs30 = DSCon['Vs30'][0]
# faulttype = DSCon['fault.type'][0]
# plot_something.distance_scaling(ML_model, DSCon_data, Vs30, Mw, faulttype,
#                                 score)

# residual = originaldata_predicted_result - original_data[1]
# residual_121 = []
# residual_199 = []
# residual_398 = []
# residual_794 = []
# residual_1000 = []

# for index, i in enumerate(np.exp(original_data[0][:, 0])):
#     if i >= 121 and i < 199:
#         residual_121.append(residual[index])
#     elif i >= 199 and i < 398:
#         residual_199.append(residual[index])
#     elif i >= 398 and i < 794:
#         residual_398.append(residual[index])
#     elif i >= 794 and i < 1000:
#         residual_794.append(residual[index])
#     elif i >= 1000:
#         residual_1000.append(residual[index])

# residual_121_mean = np.mean(residual_121)
# residual_199_mean = np.mean(residual_199)
# residual_398_mean = np.mean(residual_398)
# residual_794_mean = np.mean(residual_794)
# residual_1000_mean = np.mean(residual_1000)

# residual_121_std = np.std(residual_121)
# residual_199_std = np.std(residual_199)
# residual_398_std = np.std(residual_398)
# residual_794_std = np.std(residual_794)
# residual_1000_std = np.std(residual_1000)

# net = 50
# zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
# color_column = []

# i = 0
# while i < len(residual):  # 計算每個網格中總點數
#     x_net = (round(np.exp(original_data[0][:, 0])[i], 2) - 1e2) / (
#         (2 * 1e3 - 1e2) / net)
#     # +2:因為網格從-2開始打 10:頭減尾8-(-2) 10/net:網格間格距離 x_net:x方向第幾個網格
#     y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
#     zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
#     i += 1

# j = 0
# while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
#     x_net = (round(np.exp(original_data[0][:, 0])[j], 2) - 1e2) / (
#         (2 * 1e3 - 1e2) / net)
#     y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
#     color_column.append(zz[math.floor(x_net), math.floor(y_net)])
#     # color_column:依照資料落在哪個網格給定該資料顏色值
#     j += 1

# normalize = matplotlib.colors.Normalize(vmin=0, vmax=2000)
# colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
# newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

# plt.grid(linestyle=':', color='darkgrey')
# plt.scatter(np.exp(original_data[0][:, 0]),
#             residual,
#             c=color_column,
#             cmap=newcmp,
#             norm=normalize)
# plt.colorbar()
# plt.scatter([121, 199, 398, 794, 1000], [
#     residual_121_mean, residual_199_mean, residual_398_mean, residual_794_mean,
#     residual_1000_mean
# ],
#             marker='o',
#             color='black')
# plt.plot([121, 121], [
#     residual_121_mean + residual_121_std, residual_121_mean - residual_121_std
# ], 'black')
# plt.plot([199, 199], [
#     residual_199_mean + residual_199_std, residual_199_mean - residual_199_std
# ], 'black')
# plt.plot([398, 398], [
#     residual_398_mean + residual_398_std, residual_398_mean - residual_398_std
# ], 'black')
# plt.plot([794, 794], [
#     residual_794_mean + residual_794_std, residual_794_mean - residual_794_std
# ], 'black')
# plt.plot([1000, 1000], [
#     residual_1000_mean + residual_1000_std,
#     residual_1000_mean - residual_1000_std
# ], 'black')
# plt.plot([121, 199, 398, 794, 1000], [
#     residual_121_mean, residual_199_mean, residual_398_mean, residual_794_mean,
#     residual_1000_mean
# ], 'k--')
# plt.plot([121, 199, 398, 794, 1000], [
#     residual_121_mean + residual_121_std, residual_199_mean + residual_199_std,
#     residual_398_mean + residual_398_std, residual_794_mean + residual_794_std,
#     residual_1000_mean + residual_1000_std
# ], 'k--')
# plt.plot([121, 199, 398, 794, 1000], [
#     residual_121_mean - residual_121_std, residual_199_mean - residual_199_std,
#     residual_398_mean - residual_398_std, residual_794_mean - residual_794_std,
#     residual_1000_mean - residual_1000_std
# ], 'k--')
# plt.xscale("log")
# plt.xlim(1e2, 2 * 1e3)
# plt.ylim(-3, 3)
# plt.xlabel('Vs30(m/s)')
# plt.show()