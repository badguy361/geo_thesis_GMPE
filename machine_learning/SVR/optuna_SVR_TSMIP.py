import optuna
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.datasets
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Define an objective function to be minimized.
def objective(trial):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    svr_c = trial.suggest_loguniform('svr_c', 1e-2, 1e2)
    svr_epsilon = trial.suggest_loguniform('svr_epsilon', 1e-3, 1e3)
    regressor_obj = SVR(C=svr_c,epsilon=svr_epsilon)

    df = pd.read_csv("../../../TSMIP_FF.csv")
    df['lnVs30'] = np.log(df['Vs30'])
    df['lnRrup'] = np.log(df['Rrup'])
    x = df.loc[:,['lnVs30','MW','lnRrup']]
    y = df['PGA']

    X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = r2_score(y_val, y_pred)

    return error  # An objective value linked with the Trial object.

#ToDo
study_name = 'SVR_TSMIP_2'
study = optuna.create_study(study_name=study_name, storage="mysql://root@localhost/SVR_TSMIP",direction="maximize")  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.

print("study.best_params",study.best_params)
print("study.best_value",study.best_value)