import unittest
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from numpy.testing import assert_array_equal
import xgboost as xgb
sys.path.append("/TSMIP/machine_learning/modules")
# sys.path.append("..")
from process_train import dataprocess

model = dataprocess()


class Testprocess_train(unittest.TestCase):
    def test_preProcess(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        data['lnVs30'] = np.log(data['Vs30'])
        data['lnRrup'] = np.log(data['Rrup'])
        data['log10Vs30'] = np.log10(data['Vs30'])
        data['log10Rrup'] = np.log10(data['Rrup'])
        data['fault.type'] = data['fault.type'].str.replace("RO", "90")
        data['fault.type'] = data['fault.type'].str.replace("RV", "90")
        data['fault.type'] = data['fault.type'].str.replace("NM", "-90")
        data['fault.type'] = data['fault.type'].str.replace("NO", "-90")
        data['fault.type'] = data['fault.type'].str.replace("SS", "0")
        data['fault.type'] = pd.to_numeric(data['fault.type'])
        data[f'ln{target}(gal)'] = np.log(data[f'{target}'] * 980)
        after_process_data = data[(data["eq.type"] != "subduction interface")
                                  & (data["eq.type"] != "subduction intraslab")
                                  & (data["eq.type"] != "deep crustal")].reset_index()

        df = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        result = model.preProcess(df, target, True)

        assert after_process_data.equals(result)

    def test_splitDataset(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        after_process_data = model.preProcess(data, target, True)

        model_feature = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
        result_ori = model.splitDataset(after_process_data,
                                        f'ln{target}(gal)', True,
                                        *model_feature)

        x = after_process_data.loc[:, [x for x in model_feature]]
        y = after_process_data[target]
        x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                            y.values,
                                                            random_state=50,
                                                            train_size=0.8,
                                                            shuffle=True)
        self.assertEqual(round(x_train[0][0], 2),
                         round(result_ori[0][0][0], 2))

    def test_resetTrainTest(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        after_process_data = model.preProcess(data, target, True)
        model_feature = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
        result_ori = model.splitDataset(after_process_data,
                                        f'ln{target}(gal)', True,
                                        *model_feature)
        original_data = model.splitDataset(after_process_data, 
                                        f'ln{target}(gal)', False,
                                        *model_feature)
        new_result_ori = model.resetTrainTest(result_ori[1], result_ori[3],
                                              original_data[0], original_data[1], model_feature, f'ln{target}(gal)')

        train_SMOGN_df = pd.DataFrame(result_ori[1], columns=model_feature)
        train_SMOGN_df[target] = result_ori[3]
        train_all_df = pd.DataFrame(original_data[0], columns=model_feature)
        train_all_df[target] = original_data[1]
        merged_df = pd.merge(train_SMOGN_df, train_all_df,
                             how='right', indicator=True)
        merged_df = merged_df[(merged_df['_merge'] == 'right_only')]
        test_df = merged_df.drop(columns=['_merge'])
        test_y_array = test_df[target].to_numpy()
        test_x_array = test_df.drop(columns=[target]).to_numpy()
        np.testing.assert_array_equal(test_x_array, new_result_ori[0])
        np.testing.assert_array_equal(test_y_array, new_result_ori[1])

    def test_training(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        model = dataprocess()
        after_process_data = model.preProcess(data, target, True)
        model_feature = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
        result_ori = model.splitDataset(after_process_data,
                                        f'ln{target}(gal)', True,
                                        *model_feature)
        ML_model = model.training(
            target, "XGB", result_ori[0], result_ori[1], result_ori[2],
            result_ori[3])

        XGB_params = {'n_estimators': 893, 'eta': 0.18, 'max_depth': 30, 'gamma': 0.004,
                      'min_child_weight': 6.9, 'subsample': 0.99, 'lambda': 4.3, 'alpha': 0.24, 'n_jobs': -1}
        XGBModel = XGBRegressor(**XGB_params)
        t0 = time.time()
        grid_result = XGBModel.fit(result_ori[0], result_ori[2])
        feature_importances_test = grid_result.feature_importances_
        fit_time = time.time() - t0
        final_predict_test = XGBModel.predict(result_ori[1])
        score_test = XGBModel.score(result_ori[1], result_ori[3])
        model = XGBModel

        self.assertEqual(type(ML_model), type(model))

    def test_predictedOriginal(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        model_feature = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
        after_process_ori_data = model.preProcess(data, target, True)
        original_data = model.splitDataset(after_process_ori_data,
                                           f'ln{target}(gal)', False,
                                           *model_feature)
        booster = xgb.Booster()
        booster.load_model(f'XGB_{target}.json')
        originaldata_predicted_result = model.predicted_original(
            booster, original_data)

        predicted_result = booster.predict(xgb.DMatrix(original_data[0]))
        self.assertEqual(originaldata_predicted_result[0], predicted_result[0])


suite = unittest.TestSuite()
suite.addTest(Testprocess_train('test_preProcess'))
suite.addTest(Testprocess_train('test_splitDataset'))
suite.addTest(Testprocess_train('test_resetTrainTest'))
suite.addTest(Testprocess_train('test_training'))
suite.addTest(Testprocess_train('test_predictedOriginal'))

unittest.TextTestRunner(verbosity=2).run(suite)