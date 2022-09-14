import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

#############################################################################
# 生成随机数据
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()

# 在标签中对每50个结果标签添加噪声

y[::50] += 2 * (0.5 - rng.rand(int(X.shape[0] / 50)))

X_plot = np.linspace(0, 5, 100000)[:, None]

#############################################################################
# 训练SVR模型

#训练规模
train_size = 100
#初始化SVR
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   cv=5,
                   param_grid={
                       "C": [1e0, 1e1, 1e2, 1e3],
                       "gamma": np.logspace(-2, 2, 5)
                   },
                   n_jobs=-1)
#记录训练时间
t0 = time.time()
#训练
grid_result = svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0

# 評估，打分數
print(f"最佳準確率: {grid_result.best_score_}，最佳參數組合：{grid_result.best_params_}")
# 取得 cross validation 的平均準確率及標準差
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"平均準確率: {mean}, 標準差: {stdev}, 參數組合: {param}")

t0 = time.time()
#测试
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0

############################################################################
# 对结果进行显示
plt.scatter(X, y, c='k', label='data', zorder=1)
plt.plot(X_plot,
         y_svr,
         c='r',
         label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.figure()
