import numpy as np
import pandas as pd
import sys
import xgboost as xgb
import json
sys.path.append("..")
from modules.process_train import dataprocess
from modules.plot_figure import plot_fig
from modules.enhancement_plot import SeismicDataPlotter

targets = ["PGA", "PGV", "Sa001", "Sa002", "Sa003", "Sa004", "Sa005",
           "Sa0075", "Sa01", "Sa012", "Sa015", "Sa017", "Sa02",
           "Sa025", "Sa03", "Sa04", "Sa05", "Sa07", "Sa075", "Sa10",
           "Sa15", "Sa20", "Sa30", "Sa40", "Sa50", "Sa75", "Sa100"]
    # ? parameters
    
dataset_type = "no SMOGN" # 'cut period_shallow crustal' or 'no SMOGN'
target = "PGA"
model_type = "XGB"
plot_subset = "all dataset" # 'all dataset' or 'test subset'
Mw = 7.65
rrup = 30
Vs30 = 360
rake = 90
station_id = 350
station_id_num = 732  # station 總量
station_map = pd.read_csv("station_id_info.csv")
model_path = f'model/{dataset_type}/{model_type}_{target}.json'
model_name = [
    f'model/{dataset_type}/{model_type}_PGA.json', f'model/{dataset_type}/{model_type}_PGV.json', f'model/{dataset_type}/{model_type}_Sa001.json',
    f'model/{dataset_type}/{model_type}_Sa002.json', f'model/{dataset_type}/{model_type}_Sa003.json', f'model/{dataset_type}/{model_type}_Sa004.json',
    f'model/{dataset_type}/{model_type}_Sa005.json', f'model/{dataset_type}/{model_type}_Sa0075.json', f'model/{dataset_type}/{model_type}_Sa01.json',
    f'model/{dataset_type}/{model_type}_Sa012.json', f'model/{dataset_type}/{model_type}_Sa015.json', f'model/{dataset_type}/{model_type}_Sa017.json',
    f'model/{dataset_type}/{model_type}_Sa02.json', f'model/{dataset_type}/{model_type}_Sa025.json', f'model/{dataset_type}//{model_type}_Sa03.json',
    f'model/{dataset_type}/{model_type}_Sa04.json', f'model/{dataset_type}/{model_type}_Sa05.json', f'model/{dataset_type}/{model_type}_Sa075.json',
    f'model/{dataset_type}/{model_type}_Sa10.json', f'model/{dataset_type}/{model_type}_Sa15.json', f'model/{dataset_type}/{model_type}_Sa20.json',
    f'model/{dataset_type}/{model_type}_Sa30.json', f'model/{dataset_type}/{model_type}_Sa40.json', f'model/{dataset_type}/{model_type}_Sa50.json',
    f'model/{dataset_type}/{model_type}_Sa75.json', f'model/{dataset_type}/{model_type}_Sa100.json'
]
lowerbound = -2
higherbound = 8
model_feature = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
with open('model_info.json') as f:
    score = json.loads(f.read())
    for i in score["R2 score"]:
        if i["name"] == dataset_type:
            r2_score = i[f"{model_type}_{target}"]

# ? data preprocess
dataset = dataprocess()
TSMIP_df = pd.read_csv(
    f"../../../{dataset_type}/TSMIP_FF_period/TSMIP_FF_{target}.csv")
TSMIP_filter_df = TSMIP_df.loc[TSMIP_df['MW'] == 7.65].copy()  # filter 標準
TSMIP_Mw4_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 4)
                            & (TSMIP_df['MW'] < 5)].copy()
TSMIP_Mw5_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 5)
                            & (TSMIP_df['MW'] < 6)].copy()
TSMIP_Mw6_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 6)
                            & (TSMIP_df['MW'] < 7)].copy()
TSMIP_Mw7_df = TSMIP_df.loc[(TSMIP_df['MW'] >= 7)
                            & (TSMIP_df['MW'] < 8)].copy()

after_process_ori_data = dataset.preProcess(TSMIP_df, target, True)
after_process_ori_filter_data = dataset.preProcess(
    TSMIP_filter_df, target, True)
after_process_ori_Mw4_data = dataset.preProcess(TSMIP_Mw4_df, target, True)
after_process_ori_Mw5_data = dataset.preProcess(TSMIP_Mw5_df, target, True)
after_process_ori_Mw6_data = dataset.preProcess(TSMIP_Mw6_df, target, True)
after_process_ori_Mw7_data = dataset.preProcess(TSMIP_Mw7_df, target, True)

result_ori = dataset.splitDataset(after_process_ori_data,
                                f'ln{target}(gal)', True, *model_feature)
original_data = dataset.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, *model_feature)
original_filter_data = dataset.splitDataset(after_process_ori_filter_data, f'ln{target}(gal)',
                                            False, *model_feature)
original_Mw4_data = dataset.splitDataset(after_process_ori_Mw4_data, f'ln{target}(gal)',
                                        False, *model_feature)
original_Mw5_data = dataset.splitDataset(after_process_ori_Mw5_data, f'ln{target}(gal)',
                                        False, *model_feature)
original_Mw6_data = dataset.splitDataset(after_process_ori_Mw6_data, f'ln{target}(gal)',
                                        False, *model_feature)
original_Mw7_data = dataset.splitDataset(after_process_ori_Mw7_data, f'ln{target}(gal)',
                                        False, *model_feature)
total_Mw_data = [original_filter_data, original_Mw4_data,
                original_Mw5_data, original_Mw6_data, original_Mw7_data, original_Mw7_data]

# * SMOGN case
if dataset_type != "no SMOGN":
    TSMIP_smogn_df = pd.read_csv(
        f"../../../{dataset_type}/TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv")
    after_process_SMOGN_data = dataset.preProcess(
        TSMIP_smogn_df, target, False)
    result_SMOGN = dataset.splitDataset(after_process_SMOGN_data,
                                        f'ln{target}(gal)', True, *model_feature)
    new_result_ori = dataset.resetTrainTest(result_SMOGN[0], result_SMOGN[2],
                                            original_data[0], original_data[1], model_feature, f'ln{target}(gal)')

# ? enhancement plot
seismic_data_plotter = SeismicDataPlotter(result_ori)
seismic_data_plotter.plot_mw_distribution()
seismic_data_plotter.plot_fault_type_distribution()
seismic_data_plotter.plot_ln_pga_distribution()
eq_distribution_file = pd.read_csv("../../../TSMIP_FF_eq_distribution.csv")
# seismic_data_plotter.plot_event_distribution_map(eq_distribution_file)
# seismic_data_plotter.plot_distance_scaling(total_Mw_data, model_path)

# ? model train
# if dataset_type != "no SMOGN":
#     ML_model = dataset.training(target, f"{model_type}",
#                             result_SMOGN[0], new_result_ori[0],
#                             result_SMOGN[2], new_result_ori[1])
# else:
#     ML_model = dataset.training(target, f"{model_type}",
#                             result_ori[0], result_ori[1],
#                             result_ori[2], result_ori[3])

# ? model predicted
# booster = xgb.Booster()
# booster.load_model(model_path)
# if plot_subset == "all dataset":
#     originaldata_predicted_result = dataset.predicted_original(
#         booster, original_data)

#     plot_something = plot_fig("XGBooster", f"{model_type}", dataset_type, target,
#                               original_data[0], original_data[1], originaldata_predicted_result,
#                               after_process_ori_data, r2_score,
#                               Vs30, Mw, rrup, rake, station_id, station_id_num, station_map)

# elif plot_subset == "test subset":
#     if dataset_type != "no SMOGN": # 若是SMOGN test subset需考慮資料重複問題
#         result_ori_dataset = [new_result_ori[0], new_result_ori[1]]
#     else:
#         result_ori_dataset = [result_ori[1], result_ori[3]]
#     combined_data = np.concatenate(
#         (result_ori_dataset[0], result_ori_dataset[1][:, np.newaxis]), axis=1)
#     originaldata_predicted_result_df = pd.DataFrame(combined_data,
#                                                     columns=['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank', f'ln{target}(gal)'])
#     originaldata_predicted_result = dataset.predicted_original(
#         booster, result_ori_dataset)

#     plot_something = plot_fig("XGBooster", f"{model_type}", dataset_type, target,
#                               result_ori_dataset[0], result_ori_dataset[1], originaldata_predicted_result,
#                               originaldata_predicted_result_df, r2_score,
#                               Vs30, Mw, rrup, rake, station_id, station_id_num, station_map)
#! 檢測模型
# for i in [6,6.2,6.3,6.4,6.6,6.7,6.9,7.0,7.1,7.2,7.4,7.5,7.7,7.9,8.0]:
#     ans = booster.predict(xgb.DMatrix([(np.log(760), i, np.log(300), -90, 700)]))
#     plt.scatter(i,ans)
# plt.show()

# ? plot figure
# plot_something.data_distribution()
# plot_something.measured_predict(lowerbound, higherbound)
# plot_something.residual()
# plot_something.distance_scaling(False, total_Mw_data, model_path)
# plot_something.respond_spectrum(False, True, *model_name)
#! SHAP
# TSMIP_all_df = pd.read_csv(f"../../../TSMIP_FF.csv")
# filter = TSMIP_all_df[TSMIP_all_df['eq.type'] == "shallow crustal"].reset_index()
# station_order = filter[filter["EQ_ID"] == "1999_0920_1747_16"][["STA_Lon_X","STA_Lat_Y","STA_rank","STA_ID"]]
# index_start = station_order.index[0]
# index_end = station_order.index[-1]+1
# plot_something.explainable(station_order, model_feature,
#                             booster, index_start, index_end)
