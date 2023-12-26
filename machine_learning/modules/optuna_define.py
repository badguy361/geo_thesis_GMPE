import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


class optimize_train():
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def XGB(self, trial):
        xgb_n = trial.suggest_float('n_estimators', 100, 1000)
        xgb_dep = trial.suggest_float(
            'max_depth', 1, 50)
        xgb_eta = trial.suggest_float('eta', 0, 1)
        xgb_gamma = trial.suggest_float('gamma', 0, 1)
        xgb_min_child_weight = trial.suggest_float('min_child_weight', 0, 10)
        xgb_subsample = trial.suggest_float('subsample', 0.1, 1)
        xgb_lambda = trial.suggest_float('lambda', 0, 10)
        xgb_alpha = trial.suggest_float('alpha', 0, 10)
        XGB_params = {'eta': float(xgb_eta), 'max_depth': int(xgb_dep), 'gamma': float(xgb_gamma), 'min_child_weight': int(
            xgb_min_child_weight), 'subsample': float(xgb_subsample), 'lambda': int(xgb_lambda), 'alpha': int(xgb_alpha), 'n_jobs': -1}
        XGBModel = XGBRegressor(**XGB_params)

        grid_result = XGBModel.fit(self.x_train, self.y_train)
        print("feature importances :", grid_result.feature_importances_)
        score = XGBModel.score(self.x_test, self.y_test)
        print("test_R2_score :", score)

        # cross validation
        cv_scores = cross_val_score(XGBModel,
                                    self.x_train,
                                    self.y_train,
                                    cv=6,
                                    n_jobs=-1)
        print("cross_val R2 score:", cv_scores)

        return score


if __name__ == '__main__':
    trainer = optimize_train(1, 2, 3, 4)
    # study_name = 'XGB_TSMIP_1'
    # study = optuna.create_study(study_name=study_name,
    #                             storage="mysql://root@localhost/XGB_TSMIP",
    #                             direction="maximize")
    # study.optimize(trainer.XGB,
    #             n_trials=100)

    # print("study.best_params", study.best_params)
    # print("study.best_value", study.best_value)
    print(trainer.test)
