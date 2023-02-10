import sys
# append the path of the
# parent directory
sys.path.append("..")

from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
import pandas as pd

target = "PGA"

TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df, target)
after_process_ori_data = model.preprocess(TSMIP_df, target)
result_SMOGN = model.split_dataset(after_process_SMOGN_data,
                                   f'ln{target}(gal)', True, 'lnVs30', 'MW',
                                   'lnRrup', 'fault.type', 'STA_Lon_X',
                                   'STA_Lat_Y')
result_ori = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                 True, 'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                 'STA_Lon_X', 'STA_Lat_Y')
original_data = model.split_dataset(after_process_ori_data, f'ln{target}(gal)',
                                    False, 'lnVs30', 'MW', 'lnRrup',
                                    'fault.type', 'STA_Lon_X', 'STA_Lat_Y')
score, feature_importances, fit_time, final_predict, ML_model = model.training(
    "GBDT", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])

originaldata_predicted_result = model.predicted_original(
    ML_model, original_data)

plot_something = plot_fig("Gradient Boosting Regression", "GBDT", "SMOGN",
                          target)
plot_something.predicted_distribution(result_ori[1], result_ori[3],
                                      final_predict, fit_time, score)
plot_something.residual(original_data[0], original_data[1],
                        originaldata_predicted_result, after_process_ori_data,
                        score)
plot_something.measured_predict(original_data[1],
                                originaldata_predicted_result, score)

# minVs30 = 480
# maxVs30 = 760
# minMw = 7.0
# maxMw = 8.0
# faulttype = 1
# plot_something.distance_scaling(original_data, originaldata_predicted_result,
#                                 minVs30, maxVs30, minMw, maxMw, faulttype,
#                                 score)
