from xgboost import XGBRegressor
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.tree import plot_tree
from dtreeviz.trees import dtreeviz


def MSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse


def R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    print('R2: %2.3f' % r2)
    return r2


def two_score(y_true, y_pred):
    MSE(y_true, y_pred)  #set score here and not below if using MSE in GridCV
    score = R2(y_true, y_pred)
    return score


def two_scorer():
    return make_scorer(two_score,
                       greater_is_better=True)  # change for false if using MSE


TSMIP_smogn_df = pd.read_csv("../../../TSMIP_SMOGN_sta.csv")
TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")

TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['log10Vs30'] = np.log10(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['log10Rrup'] = np.log10(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "RO", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "RV", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "NM", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "NO", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    "SS", "3")
TSMIP_smogn_df['fault.type'] = pd.to_numeric(TSMIP_smogn_df['fault.type'])
TSMIP_smogn_df['lnPGA(gal)'] = np.log(TSMIP_smogn_df['PGA'] * 980)

TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
TSMIP_df['lnRrup'] = np.log(TSMIP_df['Rrup'])
TSMIP_df['log10Vs30'] = np.log10(TSMIP_df['Vs30'])
TSMIP_df['log10Rrup'] = np.log10(TSMIP_df['Rrup'])
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RO", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RV", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NM", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NO", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("SS", "3")
TSMIP_df['fault.type'] = pd.to_numeric(TSMIP_df['fault.type'])
TSMIP_df['lnPGA(gal)'] = np.log(TSMIP_df['PGA'] * 980)

# 對資料標準化
# df['PGA'] = (df['PGA'] - df['PGA'].mean()) / df['PGA'].std()

x_SMOGN = TSMIP_smogn_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type','STA_Lon_X','STA_Lat_Y']]
y_SMOGN = TSMIP_smogn_df['lnPGA(gal)']

x = TSMIP_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type','STA_Lon_X','STA_Lat_Y']]
y = TSMIP_df['lnPGA(gal)']

x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                    y.values,
                                                    random_state=50,
                                                    train_size=0.8,
                                                    shuffle=True)

XGBModel = XGBRegressor(
    n_estimators=1000,
    max_depth=10,
    n_jobs=-1)
t0 = time.time()
grid_result = XGBModel.fit(x_SMOGN, y_SMOGN)
print("feature importances :", grid_result.feature_importances_)
fit_time = time.time() - t0
randomForest_predict = XGBModel.predict(x_test)
# 評估，打分數
score = XGBModel.score(x_test, y_test)
print("test_R2_score :", score)

# # Cross_validation計算成績
scores = cross_val_score(XGBModel, x_train, y_train, cv=6, n_jobs=-1)
print("cross_val R2 score:", scores)
###################### visual tree #########################

viz = dtreeviz(XGBModel.estimators_[0],
               x_SMOGN,
               y_SMOGN,
               feature_names=['lnVs30', 'MW', 'lnRrup', 'fault.type'],
               title="100th decision tree - TSMIP SMOGN data")
viz.save("decision_tree_house.svg")

######################### trainsubset & testsubset distribution #########################
# 畫 Vs30 and randomForest_predict 關係圖
plt.grid(linestyle=':')
plt.scatter(x_test[:, 2],
            y_test,
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
plt.scatter(x_test[:,2], randomForest_predict,marker='o',facecolors='none',edgecolors='r', \
    label='randomForest (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnRrup')
plt.ylabel('randomForest_predict')
plt.title('Random Forest Regressor')
plt.legend()
plt.savefig(f'Rrup-RF_SMOGN_predict.png', dpi=300)
plt.show()

# 畫 Mw and randomForest_predict 關係圖
plt.grid(linestyle=':')
plt.scatter(x_test[:, 1],
            y_test,
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
plt.scatter(x_test[:,1], randomForest_predict,marker='o',facecolors='none',edgecolors='r', \
    label='randomForest (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('Mw')
plt.ylabel('randomForest_predict')
plt.title('Random Forest Regressor')
plt.legend()
plt.savefig(f'Mw-RF_SMOGN_predict.png', dpi=300)
plt.show()

######################### residual #########################
# 1. 計算Vs30_residual
residual = randomForest_predict - y_test
residual_121 = []
residual_199 = []
residual_398 = []
residual_794 = []
residual_1000 = []

for index, i in enumerate(np.exp(x_test[:, 0])):
    if i >= 121 and i < 199:
        residual_121.append(residual[index])
    elif i >= 199 and i < 398:
        residual_199.append(residual[index])
    elif i >= 398 and i < 794:
        residual_398.append(residual[index])
    elif i >= 794 and i < 1000:
        residual_794.append(residual[index])
    elif i >= 1000:
        residual_1000.append(residual[index])

residual_121_mean = np.mean(residual_121)
residual_199_mean = np.mean(residual_199)
residual_398_mean = np.mean(residual_398)
residual_794_mean = np.mean(residual_794)
residual_1000_mean = np.mean(residual_1000)

residual_121_std = np.std(residual_121)
residual_199_std = np.std(residual_199)
residual_398_std = np.std(residual_398)
residual_794_std = np.std(residual_794)
residual_1000_std = np.std(residual_1000)
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 0]),
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.scatter([121, 199, 398, 794, 1000], [
    residual_121_mean, residual_199_mean, residual_398_mean, residual_794_mean,
    residual_1000_mean
],
            marker='o',
            facecolors='none',
            edgecolors='b')
plt.plot([121, 121], [
    residual_121_mean + residual_121_std, residual_121_mean - residual_121_std
], 'b')
plt.plot([199, 199], [
    residual_199_mean + residual_199_std, residual_199_mean - residual_199_std
], 'b')
plt.plot([398, 398], [
    residual_398_mean + residual_398_std, residual_398_mean - residual_398_std
], 'b')
plt.plot([794, 794], [
    residual_794_mean + residual_794_std, residual_794_mean - residual_794_std
], 'b')
plt.plot([1000, 1000], [
    residual_1000_mean + residual_1000_std,
    residual_1000_mean - residual_1000_std
], 'b')
plt.xscale("log")
plt.xlim(1e2, 2 * 1e3)
plt.ylim(-3, 3)
plt.xlabel('Vs30(m/s)')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SMOGN SVR Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'SMOGN Vs30-RandomForest_predict_residual.png', dpi=300)
plt.show()

# 2. 計算Mw_residual
residual = randomForest_predict - y_test
residual_3_5 = []
residual_4_5 = []
residual_5_5 = []
residual_6_5 = []

for index, i in enumerate(x_test[:, 1]):
    if i >= 3.5 and i < 4.5:
        residual_3_5.append(residual[index])
    elif i >= 4.5 and i < 5.5:
        residual_4_5.append(residual[index])
    elif i >= 5.5 and i < 6.5:
        residual_5_5.append(residual[index])
    elif i >= 6.5:
        residual_6_5.append(residual[index])

residual_3_5_mean = np.mean(residual_3_5)
residual_4_5_mean = np.mean(residual_4_5)
residual_5_5_mean = np.mean(residual_5_5)
residual_6_5_mean = np.mean(residual_6_5)

residual_3_5_std = np.std(residual_3_5)
residual_4_5_std = np.std(residual_4_5)
residual_5_5_std = np.std(residual_5_5)
residual_6_5_std = np.std(residual_6_5)

plt.grid(linestyle=':', color='darkgrey')
plt.scatter(x_test[:, 1],
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.scatter([3.5, 4.5, 5.5, 6.5], [
    residual_3_5_mean, residual_4_5_mean, residual_5_5_mean, residual_6_5_mean
],
            marker='o',
            facecolors='none',
            edgecolors='b')
plt.plot([3.5, 3.5], [
    residual_3_5_mean + residual_3_5_std, residual_3_5_mean - residual_3_5_std
], 'b')
plt.plot([4.5, 4.5], [
    residual_4_5_mean + residual_4_5_std, residual_4_5_mean - residual_4_5_std
], 'b')
plt.plot([5.5, 5.5], [
    residual_5_5_mean + residual_5_5_std, residual_5_5_mean - residual_5_5_std
], 'b')
plt.plot([6.5, 6.5], [
    residual_6_5_mean + residual_6_5_std, residual_6_5_mean - residual_6_5_std
], 'b')
plt.xlim(3, 8)
plt.ylim(-3, 3)
plt.xlabel('Mw')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SMOGN RandomForest Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'SMOGN Mw-RandomForest_predict_residual.png', dpi=300)
plt.show()

# 3. 計算Rrup_residual
residual = randomForest_predict - y_test
residual_10 = []
residual_31 = []
residual_100 = []
residual_316 = []

for index, i in enumerate(np.exp(x_test[:, 2])):
    if i >= 10 and i < 31:
        residual_10.append(residual[index])
    elif i >= 31 and i < 100:
        residual_31.append(residual[index])
    elif i >= 100 and i < 316:
        residual_100.append(residual[index])
    elif i >= 316:
        residual_316.append(residual[index])

residual_10_mean = np.mean(residual_10)
residual_31_mean = np.mean(residual_31)
residual_100_mean = np.mean(residual_100)
residual_316_mean = np.mean(residual_316)

residual_10_std = np.std(residual_10)
residual_31_std = np.std(residual_31)
residual_100_std = np.std(residual_100)
residual_316_std = np.std(residual_316)
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 2]),
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.scatter(
    [10, 31, 100, 316],
    [residual_10_mean, residual_31_mean, residual_100_mean, residual_316_mean],
    marker='o',
    facecolors='none',
    edgecolors='b')
plt.plot(
    [10, 10],
    [residual_10_mean + residual_10_std, residual_10_mean - residual_10_std],
    'b')
plt.plot(
    [31, 31],
    [residual_31_mean + residual_31_std, residual_31_mean - residual_31_std],
    'b')
plt.plot([100, 100], [
    residual_100_mean + residual_100_std, residual_100_mean - residual_100_std
], 'b')
plt.plot([316, 316], [
    residual_316_mean + residual_316_std, residual_316_mean - residual_316_std
], 'b')
plt.xscale("log")
plt.xlim(5 * 1e0, 1e3)
plt.ylim(-3, 3)
plt.xlabel('Rrup(km)')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SMOGN RandomForest Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'SMOGN Rrup-RandomForest_predict_residual.png', dpi=300)
plt.show()

###################### 預測PGA和實際PGA #####################
plt.grid(linestyle=':')
plt.scatter(y_test, randomForest_predict,marker='o',facecolors='none',edgecolors='r', \
    label='Data') #迴歸線.
x_line = [-5, 10]
y_line = [-5, 10]
plt.plot(x_line, y_line, color='blue')
plt.xlabel('Measured PGA')
plt.ylabel('RandomForest_Predict PGA')
plt.ylim(-5, 10)
plt.xlim(-5, 10)
plt.title('SMOGN Measured_Predict Distribution')
plt.text(6, -2, f"R2 score = {round(score,2)}")
plt.text(6, -1, f"MAE = 0.38")
plt.legend()
plt.savefig(f'PGA_comparison_SMOGN_predict.png', dpi=300)
plt.show()