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


df = pd.read_csv("../../../TSMIP_FF.csv")
# df.head(3)

# 篩選資料 - mask方法
# df_mask_Vs30=abs(df['Vs30']-300) < 1
# df_mask_Mw=df['MW'] > 3
# df_mask_Mw=df['MW'] < 4
# filtered_df = df[df_mask_Vs30]
# filtered_df.head(10)

# 篩選資料
# filtered_df = df[(abs(df.Vs30-300)<1) & (df['MW'] > 3) & (df['MW'] < 5)]
df['lnVs30'] = np.log(df['Vs30'])
df['lnRrup'] = np.log(df['Rrup'])
df['log10Vs30'] = np.log10(df['Vs30'])
df['log10Rrup'] = np.log10(df['Rrup'])
df['fault.type'] = df['fault.type'].str.replace("RO", "1")
df['fault.type'] = df['fault.type'].str.replace("RV", "1")
df['fault.type'] = df['fault.type'].str.replace("NM", "2")
df['fault.type'] = df['fault.type'].str.replace("NO", "2")
df['fault.type'] = df['fault.type'].str.replace("SS", "3")
df['fault.type'] = pd.to_numeric(df['fault.type'])
df['lnPGA(gal)'] = np.log(df['PGA']*980)

# 對資料標準化
# df['PGA'] = (df['PGA'] - df['PGA'].mean()) / df['PGA'].std()

x = df.loc[:,['lnVs30','MW','lnRrup','fault.type']]
y = df['lnPGA(gal)'] 

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=5, train_size=0.8, shuffle=True)

# # 用paper方法當SVR參數，效果顯著
randomForestModel = RandomForestRegressor(n_estimators=1000, criterion = 'squared_error',n_jobs=-1)
t0 = time.time()
grid_result = randomForestModel.fit(x_train,y_train)
fit_time = time.time() - t0
svr_predict = randomForestModel.predict(x_test)
# 評估，打分數
score = randomForestModel.score(x_test,y_test)
print("accuracy_score : ",score)

# # Cross_validation計算成績
scores = cross_val_score(randomForestModel,x,y,cv=6,scoring=two_scorer())
print("R2 scores:",scores)
