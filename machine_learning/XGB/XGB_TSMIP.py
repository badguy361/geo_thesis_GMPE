import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import xgboost as xgb
sys.path.append("..")
from modules.process_train import dataprocess
from modules.plot_figure import plot_fig
from modules.optuna_define import optimize_train
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_optimization_history
from sklearn.model_selection import train_test_split

# ? parameters
dataset_type = "no SMOGN"
target = "PGA"
Mw = 7.65
Rrup = 40
Vs30 = 360
rake = 90
station_id = 500
station_id_num = 732  # station 總量
model_path = f'model/{dataset_type}/XGB_{target}.json'
model_name = [
    f'model/{dataset_type}/XGB_PGA.json', f'model/{dataset_type}/XGB_PGV.json', f'model/{dataset_type}/XGB_Sa001.json',
    f'model/{dataset_type}/XGB_Sa002.json', f'model/{dataset_type}/XGB_Sa003.json', f'model/{dataset_type}/XGB_Sa004.json',
    f'model/{dataset_type}/XGB_Sa005.json', f'model/{dataset_type}/XGB_Sa0075.json', f'model/{dataset_type}/XGB_Sa01.json',
    f'model/{dataset_type}/XGB_Sa012.json', f'model/{dataset_type}/XGB_Sa015.json', f'model/{dataset_type}/XGB_Sa017.json',
    f'model/{dataset_type}/XGB_Sa02.json', f'model/{dataset_type}/XGB_Sa025.json', f'model/{dataset_type}//XGB_Sa03.json',
    f'model/{dataset_type}/XGB_Sa04.json', f'model/{dataset_type}/XGB_Sa05.json', f'model/{dataset_type}/XGB_Sa075.json',
    f'model/{dataset_type}/XGB_Sa10.json', f'model/{dataset_type}/XGB_Sa15.json', f'model/{dataset_type}/XGB_Sa20.json',
    f'model/{dataset_type}/XGB_Sa30.json', f'model/{dataset_type}/XGB_Sa40.json', f'model/{dataset_type}/XGB_Sa50.json',
    f'model/{dataset_type}/XGB_Sa75.json', f'model/{dataset_type}/XGB_Sa100.json'
]
# all_SMOGN_score = {
#     'XGB_PGA': 0.87, 'XGB_PGV': 0.90, 'XGB_Sa001': 0.87, 'XGB_Sa002': 0.88,
#     'XGB_Sa003': 0.88, 'XGB_Sa004': 0.88, 'XGB_Sa005': 0.88, 'XGB_Sa0075': 0.88,
#     'XGB_Sa01': 0.88 , 'XGB_Sa012': 0.88 , 'XGB_Sa015': 0.87 , 'XGB_Sa017': 0.87 ,
#     'XGB_Sa02': 0.86, 'XGB_Sa025': 0.86, 'XGB_Sa03': 0.86, 'XGB_Sa04': 0.86,
#     'XGB_Sa05': 0.86, 'XGB_Sa075': 0.89, 'XGB_Sa10': 0.91, 'XGB_Sa15': 0.92,
#     'XGB_Sa20': 0.92, 'XGB_Sa30': 0.93, 'XGB_Sa40': 0.93, 'XGB_Sa50': 0.93,
#     'XGB_Sa75': 0.93, 'XGB_Sa100': 0.94
# }
cut_period_shallow_crustal_score = {
    'XGB_PGA': 0.88, 'XGB_PGV': 0.90, 'XGB_Sa001': 0.87, 'XGB_Sa002': 0.88,
    'XGB_Sa003': 0.89, 'XGB_Sa004': 0.88, 'XGB_Sa005': 0.88, 'XGB_Sa0075': 0.88,
    'XGB_Sa01': 0.88, 'XGB_Sa012': 0.88, 'XGB_Sa015': 0.87, 'XGB_Sa017': 0.87,
    'XGB_Sa02': 0.87, 'XGB_Sa025': 0.86, 'XGB_Sa03': 0.86, 'XGB_Sa04': 0.86,
    'XGB_Sa05': 0.86, 'XGB_Sa075': 0.89, 'XGB_Sa10': 0.91, 'XGB_Sa15': 0.92,
    'XGB_Sa20': 0.93, 'XGB_Sa30': 0.93, 'XGB_Sa40': 0.94, 'XGB_Sa50': 0.94,
    'XGB_Sa75': 0.94, 'XGB_Sa100': 0.94
}
no_SMOGN = {
    'XGB_PGA': 0.84, 'XGB_PGV': '-', 'XGB_Sa001': '-', 'XGB_Sa002': '-',
    'XGB_Sa003': '-', 'XGB_Sa004': '-', 'XGB_Sa005': '-', 'XGB_Sa0075': '-',
    'XGB_Sa01': 0.85, 'XGB_Sa012': '-', 'XGB_Sa015': '-', 'XGB_Sa017': '-',
    'XGB_Sa02': '-', 'XGB_Sa025': '-', 'XGB_Sa03': 0.82, 'XGB_Sa04': '-',
    'XGB_Sa05': '-', 'XGB_Sa075': '-', 'XGB_Sa10': 0.88, 'XGB_Sa15': '-',
    'XGB_Sa20': '-', 'XGB_Sa30': '-', 'XGB_Sa40': '-', 'XGB_Sa50': '-',
    'XGB_Sa75': '-', 'XGB_Sa100': '-'
}
lowerbound = -2
higherbound = 8
study_name = 'XGB_TSMIP_2'

# ? data preprocess
# TSMIP_smogn_df = pd.read_csv(
#     f"../../../{dataset_type}/TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(
    f"../../../{dataset_type}/TSMIP_FF_period/TSMIP_FF_{target}.csv")
TSMIP_filter_df = TSMIP_df.loc[TSMIP_df['MW'] == 7.65].copy()  # filter 標準
# Rrup range: 0.1,0.5,0.75,1,5,10,20,30,40,50,60,70,80,90,100,150,200
DSC_df = pd.read_csv(f"../../../distance_scaling_condition.csv")
TSMIP_Mw4_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 4)
                            & (TSMIP_df['MW'] < 5)].copy()
TSMIP_Mw5_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 5)
                            & (TSMIP_df['MW'] < 6)].copy()
TSMIP_Mw6_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 6)
                            & (TSMIP_df['MW'] < 7)].copy()
TSMIP_Mw7_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 7)
                            & (TSMIP_df['MW'] < 8)].copy()

dataset = dataprocess()
# after_process_SMOGN_data = dataset.preProcess(TSMIP_smogn_df, target, False)
after_process_ori_data = dataset.preProcess(TSMIP_df, target, True)
after_process_ori_filter_data = dataset.preProcess(
    TSMIP_filter_df, target, True)
after_process_ori_Mw4_data = dataset.preProcess(TSMIP_Mw4_df, target, True)
after_process_ori_Mw5_data = dataset.preProcess(TSMIP_Mw5_df, target, True)
after_process_ori_Mw6_data = dataset.preProcess(TSMIP_Mw6_df, target, True)
after_process_ori_Mw7_data = dataset.preProcess(TSMIP_Mw7_df, target, True)

model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# result_SMOGN = dataset.splitDataset(after_process_SMOGN_data,
#                                     f'ln{target}(gal)', False, *model_feture)
result_ori = dataset.splitDataset(after_process_ori_data,
                                    f'ln{target}(gal)', True, *model_feture)
original_data = dataset.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                     False, *model_feture)
original_filter_data = dataset.splitDataset(after_process_ori_filter_data, f'ln{target}(gal)',
                                            False, *model_feture)
original_Mw4_data = dataset.splitDataset(after_process_ori_Mw4_data, f'ln{target}(gal)',
                                         False, *model_feture)
original_Mw5_data = dataset.splitDataset(after_process_ori_Mw5_data, f'ln{target}(gal)',
                                         False, *model_feture)
original_Mw6_data = dataset.splitDataset(after_process_ori_Mw6_data, f'ln{target}(gal)',
                                         False, *model_feture)
original_Mw7_data = dataset.splitDataset(after_process_ori_Mw7_data, f'ln{target}(gal)',
                                         False, *model_feture)
total_Mw_data = [original_filter_data, original_Mw4_data,
                 original_Mw5_data, original_Mw6_data, original_Mw7_data]

# ? model train
#! result_ori[0](訓練資料)之shape : (29896,5) 為 29896筆 records 加上以下5個columns ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
# score, feature_importances, fit_time, final_predict, ML_model = dataset.training(
#     target, "XGB", result_ori[0], result_ori[1], result_ori[2], result_ori[3])

# ? optuna choose parameter
#! dashboard : optuna-dashboard mysql://root@localhost/XGB_TSMIP
# trainer = optimize_train(result_SMOGN[0], original_data[0], result_SMOGN[2], original_data[1])
# def objective_wrapper(trial):
#     return trainer.XGB(trial)
# study = optuna.create_study(study_name=study_name,
#                             storage="mysql://root@localhost/XGB_TSMIP",
#                             direction="maximize")
# study.optimize(objective_wrapper, n_trials=100)
# print("study.best_params", study.best_params)
# print("study.best_value", study.best_value)

# ? model predicted
booster = xgb.Booster()
booster.load_model(f"model/{dataset_type}/XGB_{target}.json")
#! residual std. 畫全部dataset
# originaldata_predicted_result = dataset.predicted_original(
#     booster, original_data)
#! residual std. 只畫test subset
result_ori_dataset = [result_ori[1], result_ori[3]]
combined_data = np.concatenate((result_ori[1], result_ori[3][:,np.newaxis]), axis=1)
originaldata_predicted_result_df = pd.DataFrame(combined_data,
                                 columns=['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank',f'ln{target}(gal)'])
originaldata_predicted_result = dataset.predicted_original(
    booster, result_ori_dataset)
#! 檢測模型
# for i in [6,6.2,6.3,6.4,6.6,6.7,6.9,7.0,7.1,7.2,7.4,7.5,7.7,7.9,8.0]:
#     ans = booster.predict(xgb.DMatrix([(np.log(760), i, np.log(300), -90, 700)]))
#     plt.scatter(i,ans)
# plt.show()

# ? plot figure(no SMOGN)
plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.data_distribution(result_SMOGN[0], result_SMOGN[1])
# mu, sigma, inter_mw_mean, inter_mw_std, intra_rrup_mean, intra_rrup_std, intra_vs30_mean, intra_vs30_std = \
#     plot_something.residual(result_ori[1], result_ori[3],
#                             originaldata_predicted_result, originaldata_predicted_result_df,
#                             no_SMOGN[f"XGB_{target}"])
# plot_something.measured_predict(result_ori_dataset[1], originaldata_predicted_result,
#                                 no_SMOGN[f"XGB_{target}"], lowerbound, higherbound)
plot_something.distance_scaling(DSC_df, Vs30, rake, station_id_num, False,
                                station_id, total_Mw_data, model_path)
# plot_something.respond_spectrum(Vs30, Mw, Rrup, rake, station_id, station_id_num,
#                                 True, False, *model_name)
#! SHAP
# TSMIP_all_df = pd.read_csv(f"../../../TSMIP_FF.csv")
# filter = TSMIP_all_df[TSMIP_all_df['eq.type'] == "shallow crustal"].reset_index()
# station_order = filter[filter["EQ_ID"] == "1999_0920_1747_16"][["STA_Lon_X","STA_Lat_Y","STA_rank","STA_ID"]]
# index_start = station_order.index[0]
# index_end = station_order.index[-1]+1
# plot_something.explainable(station_order, original_data[0], model_feture,
#                             booster, index_start, index_end)

# ? plot figure(SMOGN)
# plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
# plot_something.data_distribution(result_SMOGN[0], result_SMOGN[1])
# mu, sigma, inter_mw_mean, inter_mw_std, intra_rrup_mean, intra_rrup_std, intra_vs30_mean, intra_vs30_std = \
#     plot_something.residual(result_ori[1], result_ori[3],
#                             originaldata_predicted_result, originaldata_predicted_result,
#                             cut_period_shallow_crustal_score[f"XGB_{target}"])
# plot_something.measured_predict(original_data[1], originaldata_predicted_result,
#                                 cut_period_shallow_crustal_score[f"XGB_{target}"], lowerbound, higherbound)
# plot_something.distance_scaling(DSC_df, Vs30, rake, station_id_num, False,
#                                 station_id, total_Mw_data, model_path)
# plot_something.respond_spectrum(Vs30, Mw, Rrup, rake, station_id, station_id_num,
#                                 True, False, *model_name)
#! SHAP
# TSMIP_all_df = pd.read_csv(f"../../../TSMIP_FF.csv")
# filter = TSMIP_all_df[TSMIP_all_df['eq.type'] == "shallow crustal"].reset_index()
# station_order = filter[filter["EQ_ID"] == "1999_0920_1747_16"][["STA_Lon_X","STA_Lat_Y","STA_rank","STA_ID"]]
# index_start = station_order.index[0]
# index_end = station_order.index[-1]+1
# plot_something.explainable(station_order, original_data[0], model_feture,
#                             booster, index_start, index_end)
