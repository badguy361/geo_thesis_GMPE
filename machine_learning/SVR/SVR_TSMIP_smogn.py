import smogn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import time
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

## load data
# TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")


## conduct smogn ##這步大概要兩個小時
# TSMIP_smogn = smogn.smoter(
#     data = TSMIP_df,
#     y = "PGA"
# )
# TSMIP_smogn.to_csv("../../../TSMIP_smogn.csv",index=False)
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


TSMIP_df = pd.read_csv("../../../TSMIP_FF.csv")
TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn.csv")

TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
TSMIP_df = TSMIP_df[(TSMIP_df['Rrup'] < 1000) & (TSMIP_df['Rrup'] > 10)]
TSMIP_df = TSMIP_df[TSMIP_df['MW'] < 6.8]  # 試看看跟paper一樣的資料分布
TSMIP_df['lnRrup'] = np.log(TSMIP_df['Rrup'])
TSMIP_df['Mw_size'] = np.zeros(len(TSMIP_df['lnRrup']))
TSMIP_df['log10Vs30'] = np.log10(TSMIP_df['Vs30'])
TSMIP_df['log10Rrup'] = np.log10(TSMIP_df['Rrup'])
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RO", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RV", "1")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NM", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NO", "2")
TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("SS", "3")
TSMIP_df['fault.type'] = pd.to_numeric(TSMIP_df['fault.type'])
TSMIP_df['lnPGA(gal)'] = np.log(TSMIP_df['PGA'] * 980)

TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['Mw_size'] = np.zeros(len(TSMIP_smogn_df['lnRrup']))
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

#################################################################

TSMIP_smogn_ori_df = pd.read_csv("../../../TSMIP_smogn+oridata.csv")

TSMIP_smogn_ori_df['lnVs30'] = np.log(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['lnRrup'] = np.log(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['Mw_size'] = np.zeros(len(TSMIP_smogn_ori_df['lnRrup']))
TSMIP_smogn_ori_df['log10Vs30'] = np.log10(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['log10Rrup'] = np.log10(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("RO", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("RV", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("NM", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("NO", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df[
    'fault.type'].str.replace("SS", "3")
TSMIP_smogn_ori_df['fault.type'] = pd.to_numeric(
    TSMIP_smogn_ori_df['fault.type'])
TSMIP_smogn_ori_df['lnPGA(gal)'] = np.log(TSMIP_smogn_ori_df['PGA'] * 980)

################################## start train ######################################

# TSMIP_df原本的資料 , TSMIP_smogn_df SMOGN出來的資料 TSMIP_smogn_ori_df 融合兩個的資料
x = TSMIP_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type']]
y = TSMIP_df['lnPGA(gal)']

x_smogn_ori = TSMIP_smogn_ori_df.loc[:,
                                     ['lnVs30', 'MW', 'lnRrup', 'fault.type']]
y_smogn_ori = TSMIP_smogn_ori_df['lnPGA(gal)']

x_SMOGN = TSMIP_smogn_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type']]
y_SMOGN = TSMIP_smogn_df['lnPGA(gal)']

x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                    y.values,
                                                    random_state=60,
                                                    train_size=0.8,
                                                    shuffle=True)

svr_rbf = SVR(C=1.99, kernel='rbf', epsilon=0.001)
t0 = time.time()
grid_result = svr_rbf.fit(x_train, y_train)
fit_time = time.time() - t0
svr_predict = svr_rbf.predict(x_test)
svr_predict_train = svr_rbf.predict(x_train)
# 評估，打分數
score = svr_rbf.score(x_test, y_test)
score_train = svr_rbf.score(x_train, y_train)
print("初步得到的分數: ", score)

# # Cross_validation計算成績
scores = cross_val_score(svr_rbf, x, y, cv=6, scoring=two_scorer())
print("CV後 R2 scores:", scores)

######################### residual #########################
# 1. 計算Vs30_residual
residual = svr_predict - y_test
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 0]),
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.xscale("log")
plt.xlim(1e2, 2 * 1e3)
plt.ylim(-3, 3)
plt.xlabel('Vs30(m/s)')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SVR Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'Vs30-SVR_predict_residual.png', dpi=300)
plt.show()

# 2. 計算Mw_residual
residual = svr_predict - y_test
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(x_test[:, 1],
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.xlim(3, 8)
plt.ylim(-3, 3)
plt.xlabel('Mw')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SVR Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'Mw-SVR_predict_residual.png', dpi=300)
plt.show()

# 3. 計算Rrup_residual
residual = svr_predict - y_test
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 2]),
            residual,
            marker='o',
            facecolors='none',
            edgecolors='r')  #迴歸線
plt.xscale("log")
plt.xlim(5 * 1e0, 1e3)
plt.ylim(-3, 3)
plt.xlabel('Rrup(km)')
plt.ylabel('Residual_lnPGA(cm/s^2)')
plt.title('SVR Predict Residual R2 score: %.3f' % (score))
plt.savefig(f'Rrup-SVR_predict_residual.png', dpi=300)
plt.show()

######################### 畫svr_predict 關係圖 #########################
# 1. Vs30 and svr_predict 關係圖
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 0]),
            y_test,
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='True Values')  #數據點
plt.scatter(np.exp(x_test[:, 0]),
            svr_predict,
            marker='o',
            facecolors='none',
            edgecolors='r',
            label='Predict Values')  #迴歸線
plt.xscale("log")
plt.xlim(1e2, 2 * 1e3)
# plt.ylim(-2, 3)
plt.xlabel('Vs30(m/s)')
plt.ylabel('lnPGA(cm/s^2)')
plt.title('Vs30-SVR_predict Distribution R2 score: %.3f' % (score))
plt.legend()
plt.savefig(f'Vs30-SVR_predict.png', dpi=300)
plt.show()

# 2. Mw and svr_predict relationship
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(x_test[:, 1],
            y_test,
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='True Values')  #數據點
plt.scatter(x_test[:, 1],
            svr_predict,
            marker='o',
            facecolors='none',
            edgecolors='r',
            label='Predict Values')  #迴歸線
plt.yscale("log")
plt.xlim(3, 8)
# plt.ylim(1e-3, 1e4)
plt.xlabel('Mw')
plt.ylabel('lnPGA(cm/s^2)')
plt.title('Mw-SVR_predict Distribution R2 score: %.3f' % (score))
plt.legend()
plt.savefig(f'Mw-SVR_predict.png', dpi=300)
plt.show()

# 3. Rrup and svr_predict relationship
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(np.exp(x_test[:, 2]),
            y_test,
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='True Values')  #數據點
plt.scatter(np.exp(x_test[:, 2]),
            svr_predict,
            marker='o',
            facecolors='none',
            edgecolors='r',
            label='Predict Values')  #迴歸線
plt.xscale("log")
plt.yscale("log")
plt.xlabel('lnRrup')
plt.ylabel('lnPGA(cm/s^2)')
plt.title('Rrup-SVR_predict Distribution R2 score: %.3f' % (score))
plt.legend()
plt.savefig(f'Rrup-SVR_predict.png', dpi=300)
plt.show()

###################### 預測PGA和實際PGA #####################
# training subset
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(y_train, svr_predict_train, marker='o', facecolors='none',
            edgecolors='r',label="Data")  #迴歸線.
x = [-5, 10]
y = [-5, 10]
plt.plot(x, y, color='blue')
plt.xlabel('Measured ln(PGA)(cm/s^2)')
plt.ylabel('SVR_Predict ln(PGA)(cm/s^2)')
plt.ylim(-5, 10)
plt.xlim(-5, 10)
plt.text(6,-2,f"R2 score = {round(score_train,2)}")
plt.text(6,-1,f"MAE = 0.38")
plt.legend()
plt.title('Measured_Predict Distribution')
plt.savefig(f'Measured_Predict Training Subset.png', dpi=300)
plt.show()

# testing subset
plt.grid(linestyle=':', color='darkgrey')
plt.scatter(y_test, svr_predict, marker='o', facecolors='none',
            edgecolors='r',label="Data")  #迴歸線.
x = [-5, 10]
y = [-5, 10]
plt.plot(x, y, color='blue')
plt.xlabel('Measured ln(PGA)(cm/s^2)')
plt.ylabel('SVR_Predict ln(PGA)(cm/s^2)')
plt.ylim(-5, 10)
plt.xlim(-5, 10)
plt.text(6,-2,f"R2 score = {round(score,2)}")
plt.text(6,-1,f"MAE = 0.40")
plt.legend()
plt.title('Measured_Predict Distribution')
plt.savefig(f'Measured_Predict Testing Subset.png', dpi=300)
plt.show()