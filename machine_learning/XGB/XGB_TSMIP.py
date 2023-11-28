import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import xgboost as xgb
sys.path.append("..")
from modules.process_train import dataprocess
from modules.plot_figure import plot_fig

#? parameters
target = "Sa100"
Mw = 7
Rrup = 30
Vs30 = 760
rake = 0
station_rank = 256
model_name = [
    'model/XGB_PGA.json', 'model/XGB_PGV.json', 'model/XGB_Sa001.json',
    'model/XGB_Sa005.json', 'model/XGB_Sa01.json', 'model/XGB_Sa02.json',
    'model/XGB_Sa03.json', 'model/XGB_Sa05.json', 'model/XGB_Sa10.json',
    'model/XGB_Sa30.json', 'model/XGB_Sa40.json', 'model/XGB_Sa100.json'
]
seed = 18989
score = {
    'XGB_PGA': 0.88, 'XGB_PGV': 0.89, 'XGB_Sa001': 0.87, 'XGB_Sa005': 0.88,
    'XGB_Sa01': 0.87 , 'XGB_Sa02': 0.87, 'XGB_Sa03': 0.85, 'XGB_Sa05': 0.86,
    'XGB_Sa10': 0.90, 'XGB_Sa30': 0.93, 'XGB_Sa40': 0.94, 'XGB_Sa100': 0.93
}
lowerbound = 2
higherbound = 12

#? data preprocess
TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
model = dataprocess()
after_process_SMOGN_data = model.preProcess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preProcess(TSMIP_df, target, True)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.splitDataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)

#? model train
#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# score, feature_importances, fit_time, final_predict, ML_model = model.training(
#     target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

#? model predicted
# booster = xgb.Booster()
# booster.load_model(f'model/XGB_{target}.json')
# booster.predict(xgb.DMatrix([(np.log(760), 7, np.log(200), -45, 256)]))
# originaldata_predicted_result = model.predicted_original(
#     booster, original_data)

#? plot figure
plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.predicted_distri|bution(result_ori[1], result_ori[3],
#                                        final_predict, fit_time, score)
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score)
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score, lowerbound, higherbound)
# plot_something.distance_scaling(Vs30, Mw, Rrup, fault_type, station_rank,
#                                 original_data[0], original_data[1], booster) # 可取代
plot_something.respond_spetrum(Vs30, Mw, Rrup, rake, station_rank,
                               True, *model_name)
# plot_something.explainable(original_data[0], model_feture, booster, seed)
