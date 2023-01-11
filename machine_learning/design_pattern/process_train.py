import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.tree import plot_tree
from dtreeviz.trees import dtreeviz


class dataprocess:

    def preprocess(self, data):
        assert str(
            type(data)
        ) == "<class 'pandas.core.frame.DataFrame'>", "please input DataFrame"
        data['lnVs30'] = np.log(data['Vs30'])
        data['lnRrup'] = np.log(data['Rrup'])
        data['log10Vs30'] = np.log10(data['Vs30'])
        data['log10Rrup'] = np.log10(data['Rrup'])
        data['fault.type'] = data['fault.type'].str.replace("RO", "1")
        data['fault.type'] = data['fault.type'].str.replace("RV", "1")
        data['fault.type'] = data['fault.type'].str.replace("NM", "2")
        data['fault.type'] = data['fault.type'].str.replace("NO", "2")
        data['fault.type'] = data['fault.type'].str.replace("SS", "3")
        data['fault.type'] = pd.to_numeric(data['fault.type'])
        data['lnPGA(gal)'] = np.log(data['PGA'] * 980)
        after_process_data = data
        return after_process_data

    def split_dataset(self, after_process_data, target, split, *args):
        assert str(
            type(after_process_data)
        ) == "<class 'pandas.core.frame.DataFrame'>", "please input DataFrame"
        assert target in list(after_process_data.columns
                              ), "target is not in the DataFrame columns"
        if split == True:
            x = after_process_data.loc[:, [x for x in args]]
            y = after_process_data[target]
            x_train, x_test, y_train, y_test = train_test_split(
                x.values,
                y.values,
                random_state=50,
                train_size=0.8,
                shuffle=True)
            print("--------split_your_data----------")
            return [x_train, x_test, y_train, y_test]
        else:
            x = after_process_data.loc[:, [x for x in args]]
            y = after_process_data[target]
            print("--------not split_your_data----------")
            return [x, y]

    def training(self, model_name, x_train, x_test, y_train, y_test):
        assert model_name in [
            "SVR", "RF", "XGB", "GBDT", "DNN", "Ada"
        ], "please choose one method in [SVR, RF, XGB, GBDT, DNN, Ada] or add the new method by yourself"
        if model_name == "RF":
            rfr_params = {
                'n_estimators': 100,
                'criterion': 'squared_error',
                'bootstrap': True,
                'oob_score': True,
                #   verbose=True,
                'n_jobs': -1
            }
            randomForestModel = RandomForestRegressor(**rfr_params)
            t0 = time.time()
            grid_result = randomForestModel.fit(x_train, y_train)
            feature_importances = grid_result.feature_importances_
            print("oob_score :", grid_result.oob_score_)
            print("feature importances :", grid_result.feature_importances_)
            fit_time = time.time() - t0
            final_predict = randomForestModel.predict(x_test)
            # 評估，打分數
            score = randomForestModel.score(x_test, y_test)
            print("test_R2_score :", score)

            # cross validation
            cv_scores = cross_val_score(randomForestModel,
                                        x_train,
                                        y_train,
                                        cv=6,
                                        n_jobs=-1)
            print("cross_val R2 score:", cv_scores)

        elif model_name == "GBDT":
            gbr_params = {
                'n_estimators': 1000,
                'max_depth': 3,
                'min_samples_split': 5,
                'learning_rate': 0.5,
                'loss': 'squared_error'
            }
            GradientBoostingModel = GradientBoostingRegressor(**gbr_params)
            t0 = time.time()
            grid_result = GradientBoostingModel.fit(x_train, y_train)
            feature_importances = grid_result.feature_importances_
            print("feature importances :", grid_result.feature_importances_)
            fit_time = time.time() - t0
            final_predict = GradientBoostingModel.predict(x_test)
            # 評估，打分數
            score = GradientBoostingModel.score(x_test, y_test)
            print("test_R2_score :", score)

            # cross validation
            cv_scores = cross_val_score(GradientBoostingModel,
                                        x_train,
                                        y_train,
                                        cv=6,
                                        n_jobs=-1)
            print("cross_val R2 score:", cv_scores)

        elif model_name == "Ada":
            Ada_params = {
                'n_estimators': 1000,
                'learning_rate': 0.005,
                'loss': 'exponential'
            }
            AdaBoostModel = AdaBoostRegressor(**Ada_params)
            t0 = time.time()
            grid_result = AdaBoostModel.fit(x_train, y_train)
            feature_importances = grid_result.feature_importances_
            print("feature importances :", grid_result.feature_importances_)
            fit_time = time.time() - t0
            final_predict = AdaBoostModel.predict(x_test)
            # 評估，打分數
            score = AdaBoostModel.score(x_test, y_test)
            print("test_R2_score :", score)

            # cross validation
            cv_scores = cross_val_score(AdaBoostModel,
                                        x_train,
                                        y_train,
                                        cv=6,
                                        n_jobs=-1)
            print("cross_val R2 score:", cv_scores)
        
        elif model_name == "XGB":
            XGB_params = {
                'n_estimators': 3000,
                'max_depth': 20,
                'n_jobs':-1
            }
            XGBModel = XGBRegressor(**XGB_params)
            t0 = time.time()
            grid_result = XGBModel.fit(x_train, y_train)
            feature_importances = grid_result.feature_importances_
            print("feature importances :", grid_result.feature_importances_)
            fit_time = time.time() - t0
            final_predict = XGBModel.predict(x_test)
            # 評估，打分數
            score = XGBModel.score(x_test, y_test)
            print("test_R2_score :", score)

            # cross validation
            cv_scores = cross_val_score(XGBModel,
                                        x_train,
                                        y_train,
                                        cv=6,
                                        n_jobs=-1)
            print("cross_val R2 score:", cv_scores)

        else:
            pass

        return score, feature_importances, fit_time, final_predict


if __name__ == '__main__':
    TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn_sta.csv")
    TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")
    model = dataprocess()
    after_process_data = model.preprocess(TSMIP_df)
    result_list = model.split_dataset(TSMIP_df, 'lnPGA(gal)', True,
                                      'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                      'STA_Lon_X', 'STA_Lat_Y')
    score, feature_importances, fit_time, final_predict = model.training(
        "GBDT", result_list[0], result_list[1], result_list[2], result_list[3])
