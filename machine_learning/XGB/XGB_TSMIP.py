import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import xgboost as xgb
sys.path.append("..")
from modules.process_train import dataprocess
from modules.plot_figure import plot_fig
from modules.optuna_define import optimize_train
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances


#? parameters
target = "PGA"
Mw = 7
Rrup = 30
Vs30 = 360
rake = 0
station_rank = 265
station_id_num = 732 # station 總量
model_name = [
    'model/XGB_PGA.json', 'model/XGB_PGV.json', 'model/XGB_Sa001.json',
    'model/XGB_Sa005.json', 'model/XGB_Sa01.json', 'model/XGB_Sa02.json',
    'model/XGB_Sa03.json', 'model/XGB_Sa05.json', 'model/XGB_Sa10.json',
    'model/XGB_Sa30.json', 'model/XGB_Sa40.json', 'model/XGB_Sa100.json'
]
score = {
    'XGB_PGA': 0.88, 'XGB_PGV': 0.89, 'XGB_Sa001': 0.87, 'XGB_Sa005': 0.88,
    'XGB_Sa01': 0.87 , 'XGB_Sa02': 0.87, 'XGB_Sa03': 0.85, 'XGB_Sa05': 0.86,
    'XGB_Sa10': 0.90, 'XGB_Sa30': 0.93, 'XGB_Sa40': 0.94, 'XGB_Sa100': 0.93
}
lowerbound = -2
higherbound = 8
seed = 18989
study_name = 'XGB_TSMIP_2'

#? data preprocess
TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_period/TSMIP_FF_{target}.csv")
TSMIP_filter_df = TSMIP_df.loc[TSMIP_df['MW'] > 7].copy() # filter 標準
DSC_df = pd.read_csv(f"../../../distance_scaling_condition.csv") # Rrup range: 0.1,0.5,0.75,1,5,10,20,30,40,50,60,70,80,90,100,150,200
model = dataprocess()
after_process_SMOGN_data = model.preProcess(TSMIP_smogn_df, target, False)
after_process_ori_data = model.preProcess(TSMIP_df, target, True)
after_process_ori_filter_data = model.preProcess(TSMIP_filter_df, target, True)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
result_SMOGN = model.splitDataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, *model_feture)
result_ori = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, *model_feture)
original_data = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feture)
original_filter_data = model.splitDataset(after_process_ori_filter_data, f'ln{target}(gal)',
                                    False, *model_feture)

#? model train
#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# score, feature_importances, fit_time, final_predict, ML_model = model.training(
#     target, "XGB", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

#? optuna choose parameter
#! dashboard : optuna-dashboard mysql://root@localhost/XGB_TSMIP
# trainer = optimize_train(result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])
# def objective_wrapper(trial):
#     return trainer.XGB(trial)
# study = optuna.create_study(study_name=study_name,
#                             storage="mysql://root@localhost/XGB_TSMIP",
#                             direction="maximize")
# study.optimize(objective_wrapper, n_trials=100)
# print("study.best_params", study.best_params)
# print("study.best_value", study.best_value)

#? model predicted
booster = xgb.Booster()
booster.load_model(f'model/XGB_{target}.json')
# booster.predict(xgb.DMatrix([(np.log(760), 7, np.log(200), -45, 256)]))
originaldata_predicted_result = model.predicted_original(
    booster, original_data)

#? plot figure
plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.data_distribution(original_data[0], original_data[1])
# plot_something.residual(original_data[0], original_data[1],
#                         originaldata_predicted_result, after_process_ori_data,
#                         score[f"XGB_{target}"])
# plot_something.measured_predict(original_data[1], originaldata_predicted_result, score[f"XGB_{target}"], lowerbound, higherbound)
plot_something.distance_scaling(DSC_df, station_id_num, False,
                                original_filter_data[0], original_filter_data[1], "model/XGB_PGA.json")
# plot_something.respond_spectrum(Vs30, Mw, Rrup, rake, station_rank,
#                                False, *model_name)
# plot_something.explainable(original_data[0], model_feture, booster, seed)
