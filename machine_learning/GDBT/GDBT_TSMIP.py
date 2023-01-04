import sys
# append the path of the
# parent directory
sys.path.append("..")

from design_pattern.design_pattern import dataprocess
import pandas as pd

TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn_sta.csv")
TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")
model = dataprocess()
after_process_SMOGN_data = model.preprocess(TSMIP_smogn_df)
after_process_ori_data = model.preprocess(TSMIP_df)
result_SMOGN = model.split_dataset(TSMIP_smogn_df, 'lnPGA(gal)', True, 'lnVs30',
                                  'MW', 'lnRrup', 'fault.type', 'STA_Lon_X',
                                  'STA_Lat_Y')
result_ori = model.split_dataset(TSMIP_df, 'lnPGA(gal)', True, 'lnVs30',
                                  'MW', 'lnRrup', 'fault.type', 'STA_Lon_X',
                                  'STA_Lat_Y')
score, feature_importances, fit_time, final_predict = model.training(
    "GDBT", result_SMOGN[0], result_ori[1], result_SMOGN[2], result_ori[3])
