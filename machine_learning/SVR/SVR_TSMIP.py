import sys
# append the path of the
# parent directory
sys.path.append("..")

from design_pattern.process_train import dataprocess
from design_pattern.plot_figure import plot_fig
import pandas as pd

TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn_sta.csv")
TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df)
after_process_ori_data = model.preprocess(TSMIP_df)
result_SMOGN = model.split_dataset(TSMIP_smogn_df, 'lnPGA(gal)', True, 'lnVs30',
                                  'MW', 'lnRrup', 'fault.type')
result_ori = model.split_dataset(TSMIP_df, 'lnPGA(gal)', True, 'lnVs30',
                                  'MW', 'lnRrup', 'fault.type')
score, feature_importances, fit_time, final_predict = model.training(
    "SVR", result_ori[0], result_ori[1], result_ori[2], result_ori[3])

plot_something = plot_fig("Support Vector Regression","SVR","TSMIP")
plot_something.train_test_distribution(result_ori[1], result_ori[3], final_predict, fit_time, score)
plot_something.residual(result_ori[1], result_ori[3], final_predict, score)
plot_something.measured_predict(result_ori[3], final_predict, score)
c = result_ori[1].transpose(1, 0)
plot_something.distance_scaling(c[2], final_predict, score)
