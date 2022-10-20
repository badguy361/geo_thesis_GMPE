import optuna
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.datasets
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from optuna.visualization import plot_optimization_history
import plotly
from optuna.visualization import plot_param_importances


# Define an objective function to be minimized.
def objective(trial):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    svr_c = trial.suggest_float('svr_c', 1e-3, 2)
    svr_epsilon = trial.suggest_float('svr_epsilon', 1e-3, 1e-2)
    svr_gamma = trial.suggest_float('svr_gamma', 1e-3, 10)
    regressor_obj = SVR(C=svr_c, epsilon=svr_epsilon, gamma=svr_gamma)

    # TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn.csv")
    TSMIP_df = pd.read_csv("../../../TSMIP_FF.csv")

    # TSMIP_smogn_df['lnVs30'] = np.log(TSMIP_smogn_df['Vs30'])
    # TSMIP_smogn_df['lnRrup'] = np.log(TSMIP_smogn_df['Rrup'])
    # TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    #     "RO", "1")
    # TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    #     "RV", "1")
    # TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    #     "NM", "2")
    # TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    #     "NO", "2")
    # TSMIP_smogn_df['fault.type'] = TSMIP_smogn_df['fault.type'].str.replace(
    #     "SS", "3")
    # TSMIP_smogn_df['fault.type'] = pd.to_numeric(TSMIP_smogn_df['fault.type'])
    # TSMIP_smogn_df['lnPGA(gal)'] = np.log(TSMIP_smogn_df['PGA'] * 980)

    TSMIP_df['lnVs30'] = np.log(TSMIP_df['Vs30'])
    TSMIP_df['lnRrup'] = np.log(TSMIP_df['Rrup'])
    TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RO", "1")
    TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("RV", "1")
    TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NM", "2")
    TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("NO", "2")
    TSMIP_df['fault.type'] = TSMIP_df['fault.type'].str.replace("SS", "3")
    TSMIP_df['fault.type'] = pd.to_numeric(TSMIP_df['fault.type'])
    TSMIP_df['lnPGA(gal)'] = np.log(TSMIP_df['PGA'] * 980)

    x = TSMIP_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type']]
    y = TSMIP_df['lnPGA(gal)']

    # x_SMOGN = TSMIP_smogn_df.loc[:, ['lnVs30', 'MW', 'lnRrup', 'fault.type']]
    # y_SMOGN = TSMIP_smogn_df['lnPGA(gal)']

    X_train, X_val, y_train, y_val = train_test_split(x,
                                                      y,
                                                      random_state=0,
                                                      train_size=0.8)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = r2_score(y_val, y_pred)

    return error  # An objective value linked with the Trial object.


#ToDo
#SVR_TSMIP_4 新增SMOGN資料
#SVR_TSMIP_5 SMOGN資料當訓練集 原本資料切20%當測試集
#SVR_TSMIP_6 從頭來過，把y改成ln(PGA)，找TSMIP的最佳解 -> C=1.99 epsilon=0.001
#SVR_TSMIP_7 y改成ln(PGA)，SMOGN資料當訓練集 原本資料切20%當測試集，並新增fault.type參數 -> 成效反而比較差
#SVR_TSMIP_8 y改成ln(PGA)，原本資料當訓練集 原本資料切20%當測試集，並新增gamma讓他跑分
#SVR_TSMIP_9 擴大gamma範圍讓他跑分
study_name = 'SVR_TSMIP_9'
study = optuna.create_study(study_name=study_name,
                            storage="mysql://root@localhost/SVR_TSMIP",
                            direction="maximize")  # Create a new study.
study.optimize(objective,
               n_trials=100)  # Invoke optimization of the objective function.

print("study.best_params", study.best_params)
print("study.best_value", study.best_value)

loaded_study = optuna.load_study(study_name=study_name,
                                 storage="mysql://root@localhost/SVR_TSMIP")
plotly_config = {"staticPlot": True}

fig = plot_optimization_history(loaded_study)
fig.show(config=plotly_config)

fig = plot_param_importances(loaded_study)
fig.show(config=plotly_config)