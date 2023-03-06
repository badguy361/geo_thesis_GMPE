import xgboost
import shap

# 載入示例數據集
X,y = shap.datasets.diabetes()
# 定義 XGBoost 模型
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
# 創建一個 SHAP 解釋器
explainer = shap.Explainer(model)
# 使用 SHAP 解釋器解釋數據
shap_values = explainer(X)
# 打印每個特徵的 SHAP 值
shap.summary_plot(shap_values, X)