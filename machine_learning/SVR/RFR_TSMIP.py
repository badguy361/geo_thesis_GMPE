from sklearn.ensemble import RandomForestRegressor
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


TSMIP_smogn_df = pd.read_csv("../../../TSMIP_SMOGN.csv")
TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")

TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['log10Vs30'] = np.log10(TSMIP_smogn_df['Vs30'])
TSMIP_smogn_df['log10Rrup'] = np.log10(TSMIP_smogn_df['Rrup'])
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("RO", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("RV", "1")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("NM", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("NO", "2")
TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace("SS", "3")
TSMIP_smogn_df['fault.type'] = pd.to_numeric(TSMIP_smogn_df['fault.type'])
TSMIP_smogn_df['lnPGA(gal)'] = np.log(TSMIP_smogn_df['PGA']*980)

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
TSMIP_df['lnPGA(gal)'] = np.log(TSMIP_df['PGA']*980)

# 對資料標準化
# df['PGA'] = (df['PGA'] - df['PGA'].mean()) / df['PGA'].std()

x_SMOGN = TSMIP_smogn_df.loc[:,['lnVs30','MW','lnRrup','fault.type']]
y_SMOGN = TSMIP_smogn_df['lnPGA(gal)'] 

x = TSMIP_smogn_df.loc[:,['lnVs30','MW','lnRrup','fault.type']]
y = TSMIP_smogn_df['lnPGA(gal)'] 

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=50, train_size=0.8, shuffle=True)

randomForestModel = RandomForestRegressor(n_estimators=1000, criterion = 'squared_error',max_depth=20, n_jobs=-1)
t0 = time.time()
grid_result = randomForestModel.fit(x_SMOGN,y_SMOGN)
fit_time = time.time() - t0
randomForest_predict = randomForestModel.predict(x)
# 評估，打分數
score = randomForestModel.score(x,y)
print("accuracy_score : ",score)

# # Cross_validation計算成績
scores = cross_val_score(randomForestModel,x,y,cv=3,n_jobs=-1)
print("R2 scores:",scores)


plt.grid(linestyle=':')
plt.scatter(y, randomForest_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線.
x=[0,2,4,6,8,10]
y=[0,2,4,6,8,10]
plt.plot(x, y, color='blue')
plt.xlabel('Measured PGA')
plt.ylabel('randomForest_predict PGA')
plt.ylim(0,10)
plt.xlim(0,10)
plt.title('Random Forest Regressor')
plt.legend()
plt.savefig(f'PGA_comparison_SMOGN_predict.png',dpi=300)
plt.show()