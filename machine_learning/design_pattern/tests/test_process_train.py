import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import pickle
from numpy.testing import assert_array_equal

sys.path.append("/TSMIP/machine_learning/design_pattern")
from process_train import dataprocess

model = dataprocess()


class Testprocess_train(unittest.TestCase):
    def test_preprocess(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
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
        data[f'ln{target}(gal)'] = np.log(data[f'{target}'] * 980)
        after_process_data = data[(data["eq.type"] != "subduction interface")
                                  & (data["eq.type"] != "subduction intraslab")
                                  & (data["eq.type"] != "deep crustal")]

        df = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        result = model.preprocess(df, target, True)

        assert after_process_data.equals(result)

    def test_split_dataset(self):
        target = "PGA"
        data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
        after_process_data = model.preprocess(data, target, True)

        model_feture = [
            'lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank'
        ]
        result_ori = model.split_dataset(after_process_data,
                                         f'ln{target}(gal)', True,
                                         *model_feture)

        x = after_process_data.loc[:, [x for x in model_feture]]
        y = after_process_data[target]
        x_train, x_test, y_train, y_test = train_test_split(x.values,
                                                            y.values,
                                                            random_state=50,
                                                            train_size=0.8,
                                                            shuffle=True)
        assert self.assertEqual(6.06, 6.06) 

    # def test_training(self):
    #     target = "PGA"
    #     data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
    #     model = dataprocess()
    #     after_process_data = model.preprocess(data, target, True)
    #     model_feture = [
    #         'lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank'
    #     ]
    #     result_ori = model.split_dataset(after_process_data,
    #                                      f'ln{target}(gal)', True,
    #                                      *model_feture)
    #     score_model, feature_importances_model, fit_time, final_predict_model, ML_model = model.training(
    #         target, "XGB", result_ori[0], result_ori[1], result_ori[2],
    #         result_ori[3])

    #     XGB_params = {'n_estimators': 1000, 'max_depth': 10, 'n_jobs': -1}
    #     XGBModel = XGBRegressor(**XGB_params)
    #     t0 = time.time()
    #     grid_result = XGBModel.fit(result_ori[0], result_ori[2])
    #     feature_importances_test = grid_result.feature_importances_
    #     fit_time = time.time() - t0
    #     final_predict_test = XGBModel.predict(result_ori[1])
    #     score_test = XGBModel.score(result_ori[1], result_ori[3])
    #     model = XGBModel

    #     assert self.assertEqual(score_model, score_test)

    # def test_predicted_original(self):
    #     target = "PGA"
    #     data = pd.read_csv("/TSMIP/TSMIP_FF_test.csv")
    #     model_feture = [
    #         'lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank'
    #     ]
    #     after_process_ori_data = model.preprocess(data, target, True)
    #     original_data = model.split_dataset(after_process_ori_data,
    #                                         f'ln{target}(gal)', False,
    #                                         *model_feture)
    #     ML_model = pickle.load(
    #         open(f'/TSMIP/machine_learning/XGB/model/XGB_{target}.pkl', 'rb'))
    #     originaldata_predicted_result = model.predicted_original(
    #         ML_model, original_data)

    #     predicted_result = ML_model.predict(original_data[0])
    #     assert self.assertEqual(originaldata_predicted_result[0],
    #                               predicted_result[0])


suite = unittest.TestSuite()
suite.addTest(Testprocess_train('test_preprocess'))
suite.addTest(Testprocess_train('test_split_dataset'))
# suite.addTest(Testprocess_train('test_training'))
# suite.addTest(Testprocess_train('test_predicted_original'))

unittest.TextTestRunner(verbosity=2).run(suite)