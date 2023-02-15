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
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score)

Mw = DSCon['MW'][0]
Vs30 = DSCon['Vs30'][0]
faulttype = DSCon['fault.type'][0]
plot_something.distance_scaling(ML_model, DSCon_data, Vs30, Mw, faulttype,
                                score)

net = 50
x = np.linspace(-2, 8, net)
y = np.linspace(-2, 8, net)

xx, yy = np.meshgrid(x, y)
zz = np.array([0] * net * net).reshape(net, net)
color_column = []

i = 0
while i < len(original_data[1]):
    x_net = (round(original_data[1][i], 2) + 2) / (10 / net)
    y_net = (round(originaldata_predicted_result[i], 2) + 2) / (10 / net)
    # print(math.floor(x_net),math.floor(y_net))
    print(i)
    zz[math.floor(x_net), math.floor(y_net)] += 1
    i += 1

j = 0 
while j < len(original_data[1]):
    x_net = (round(original_data[1][j], 2) + 2) / (10 / net)
    y_net = (round(originaldata_predicted_result[j], 2) + 2) / (10 / net)
    color_column.append(zz[math.floor(x_net), math.floor(y_net)])
    j += 1

normalize = matplotlib.colors.Normalize(vmin=0, vmax=2000)
colorlist = [
    "deepskyblue", "aquamarine","yellow", "orange", "red"
]
newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
x_line = [-2, 8]
y_line = [-2, 8]
plt.plot(x_line, y_line, 'r--', alpha = 0.5)
plt.grid(linestyle=':')
plt.scatter(
    original_data[1],
    originaldata_predicted_result,
    c=color_column,
    cmap=newcmp,
    norm=normalize
)
plt.colorbar()
plt.show()