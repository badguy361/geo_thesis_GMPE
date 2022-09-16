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
# x = df.loc[:,['Vs30','MW','Rrup']]
x = df.loc[:,['lnVs30','MW','lnRrup']]
y = df['PGA']

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=10, train_size=0.8)

# follow the paper
C = max(abs(y_train.mean() + 3 * y_train.std()),abs(y_train.mean() - 3 * y_train.std()))

m = y_train.max()-y_train.min()
k = 17
p = m * (0.3 ** (1/k))
gamma = 1 / (2 * (p**2))
epsilon = 3* 0.18 * (np.log(29896)/29896)**(1/2)

# 用paper方法當SVR參數，效果顯著
svr_rbf = SVR(C=C, kernel='rbf', gamma=gamma, epsilon=epsilon)
# Cross_validation計算成績
scores = cross_val_score(svr_rbf,x,y,cv=5,scoring=two_scorer())
print(scores)

# GridSearch找最佳參數
svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   cv=6,
                   param_grid={
                       "C": [1e0, 1e1, 1e2, 1e3],
                       "gamma": np.logspace(-10, -5, 6)
                    #    "epsilon": [1e-2]
                   },
                   scoring=two_scorer(),
                   n_jobs=-1)
t0 = time.time()
grid_result = svr_rbf.fit(x_train,y_train)
fit_time = time.time() - t0
svr_predict = svr_rbf.predict(x_test)

# 評估，打分數
score = svr_rbf.score(x_test,y_test)
print("accuracy_score : ",score)

# GridSearchCV打分數
print(f"最佳準確率: {grid_result.best_score_}，最佳參數組合：{grid_result.best_params_}")
# 取得 cross validation 的平均準確率及標準差
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"平均準確率: {mean}, 標準差: {stdev}, 參數組合: {param}")


# 依照欄位大小排序，為了畫圖好看
x_test_tmp_df = pd.DataFrame(x_test)
x_test_tmp_df.sort_values(by=[1],inplace=True) # inplace 會改變id
x_test = x_test_tmp_df.to_numpy()

# plt Vs30 and svr_predict relationship
plt.scatter(x_test[:,0], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,0], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnVs30')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Vs30-SVR_predict.png',dpi=300)
plt.show()

# plt Mw and svr_predict relationship
plt.scatter(x_test[:,1], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,1], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('Mw')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Mw-SVR_predict.png',dpi=300)
plt.show()

# plt Rrup and svr_predict relationship
plt.scatter(x_test[:,2], y_test, marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.scatter(x_test[:,2], svr_predict,marker='o',facecolors='none',edgecolors='r', \
    label='SVR (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
plt.xlabel('lnRrup')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.savefig(f'Rrup-SVR_predict.png',dpi=300)
plt.show()