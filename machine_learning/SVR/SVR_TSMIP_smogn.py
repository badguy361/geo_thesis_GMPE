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
def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print('MSE: %2.3f' % mse)
    return mse

def R2(y_true,y_pred):    
    r2 = r2_score(y_true, y_pred)
    print('R2: %2.3f' % r2)
    return r2

def two_score(y_true,y_pred):    
    MSE(y_true,y_pred) #set score here and not below if using MSE in GridCV
    score = R2(y_true,y_pred)
    return score

def two_scorer():
    return make_scorer(two_score, greater_is_better=True) # change for false if using MSE


TSMIP_df = pd.read_csv("../../../TSMIP_FF.csv")
TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn.csv")

TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
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

TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['Mw_size'] = np.zeros(len(TSMIP_smogn_df['lnRrup']))
TSMIP_smogn_df['log10Vs30'] = np.log10(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['log10Rrup'] = np.log10(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("RO", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("RV", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("NM", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("NO", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("SS", "3")
TSMIP_smogn_df['fault.type'] = pd.to_numeric(TSMIP_smogn_df['fault.type'])

#################################################################

TSMIP_smogn_ori_df = pd.read_csv("../../../TSMIP_smogn+oridata.csv")

TSMIP_smogn_ori_df['lnVs30'] = np.log(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['lnRrup'] = np.log(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['Mw_size'] = np.zeros(len(TSMIP_smogn_ori_df['lnRrup']))
TSMIP_smogn_ori_df['log10Vs30'] = np.log10(TSMIP_smogn_ori_df['Vs30'])
TSMIP_smogn_ori_df['log10Rrup'] = np.log10(TSMIP_smogn_ori_df['Rrup'])
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df['fault.type'].str.replace("RO", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df['fault.type'].str.replace("RV", "1")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df['fault.type'].str.replace("NM", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df['fault.type'].str.replace("NO", "2")
TSMIP_smogn_ori_df['fault.type'] = TSMIP_smogn_ori_df['fault.type'].str.replace("SS", "3")
TSMIP_smogn_ori_df['fault.type'] = pd.to_numeric(TSMIP_smogn_ori_df['fault.type'])

# 對資料標準化
# df['PGA'] = (df['PGA'] - df['PGA'].mean()) / df['PGA'].std()

# x_test = TSMIP_df.loc[:,['lnVs30','MW','lnRrup']]
# y_test = TSMIP_df['PGA']

x = TSMIP_smogn_ori_df.loc[:,['lnVs30','MW','lnRrup']]
y = TSMIP_smogn_ori_df['PGA']

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=10, train_size=0.8)

svr_rbf = SVR(C=0.46, kernel='rbf', epsilon=0.0008)
t0 = time.time()
grid_result = svr_rbf.fit(x,y)
fit_time = time.time() - t0
svr_predict = svr_rbf.predict(x_test)
# 評估，打分數
score = svr_rbf.score(x_test,y_test)
print("accuracy_score : ",score)

# # Cross_validation計算成績
scores = cross_val_score(svr_rbf,x,y,cv=5,scoring=two_scorer())
print("R2 scores:",scores)


# ################### Rrup Mw distribution ###########################

# plt.grid(color='gray', linestyle = '--',linewidth=0.5)
# plt.scatter(TSMIP_df['Rrup'], TSMIP_df['MW'], marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
# plt.xscale("log")
# plt.xlim(1e-1,1e3)
# plt.xlabel('Rrup')
# plt.ylabel('Mw')
# plt.title('TSMIP_SMOGN Data distribution')
# plt.legend()
# plt.savefig(f'TSMIP_SMOGN Data distribution.png',dpi=300)
# plt.show()

# ################### Vs30 Number of ground motion ###########################

# plt.grid(color='gray', linestyle = '--',linewidth=0.5)
# plt.hist(TSMIP_df['Vs30'], bins=20, edgecolor="yellow", color="green")
# plt.xlabel('Vs30(m/s)')
# plt.ylabel('Number of record')
# plt.title('TSMIP_SMOGN Vs30 distribution')
# plt.legend()
# plt.savefig(f'TSMIP_SMOGN Vs30 distribution.png',dpi=300)
# plt.show()



########################## 計算Vs30_residual #########################
residual = svr_predict - y_test
plt.grid(linestyle=':')
plt.scatter(x_test[:,0], residual ,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnVs30(km)')
plt.ylabel('Predict_PGA(g)')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Vs30-SVR_SMOGN_predict_residual.png',dpi=300)
plt.show()

# # 計算Mw_residual
plt.grid(linestyle=':')
residual = svr_predict - y_test
plt.scatter(x_test[:,1], residual ,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('Mw')
plt.ylabel('Predict_PGA(g)')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Mw-SVR_SMOGN_predict_residual.png',dpi=300)
plt.show()

# # 計算Rrup_residual
plt.grid(linestyle=':')
residual = svr_predict - y_test
plt.scatter(x_test[:,2], residual ,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnRrup(km)')
plt.ylabel('Predict_PGA(g)')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Rrup-SVR_SMOGN_predict_residual.png',dpi=300)
plt.show()


# 畫 Vs30 and svr_predict 關係圖
plt.grid(linestyle=':')
plt.scatter(x_test[:,0], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,0], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnVs30')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Vs30-SVR_SMOGN_predict.png',dpi=300)
plt.show()

# plt Mw and svr_predict relationship
plt.grid(linestyle=':')
plt.scatter(x_test[:,1], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,1], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('Mw')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Mw-SVR_SMOGN_predict.png',dpi=300)
plt.show()

# plt Rrup and svr_predict relationship
plt.grid(linestyle=':')
plt.scatter(x_test[:,2], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,2], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnRrup')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Rrup-SVR_SMOGN_predict.png',dpi=300)
plt.show()