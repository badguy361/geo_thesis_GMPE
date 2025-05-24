import numpy as np
import pandas as pd
import xgboost as xgb

# TODO: 記得改成對應路徑
# 路徑設定 
STATION_INFO_CSV = "machine_learning/XGB/station_id_info.csv"
MODEL_PATH = "machine_learning/XGB/model/no SMOGN/XGB_PGA.json"

# 載入模型與站點資訊
model = xgb.Booster()
model.load_model(MODEL_PATH)
station_info_df = pd.read_csv(STATION_INFO_CSV)

# 標準參考點座標（longitude, latitude）
STANDARD_POINT = [119.5635611, 21.90093889]

# 輸入地震參數
vs30 = 760                # 地盤剪力波速 (m/s)
magnitude = 7             # 地震規模
rupture_distance = 10     # 震源至站點距離 (km)
fault_rake = 90            # 斷層滑移角度

# 目標站點資訊（以 KAU084 為例）
target_lon = 120.3691
target_lat = 22.3467

# 計算站點與標準點的距離
lat_diff_km = (target_lat - STANDARD_POINT[1]) * 110
lon_diff_km = (target_lon - STANDARD_POINT[0]) * 101
distance_to_standard = np.sqrt(lat_diff_km**2 + lon_diff_km**2)

# 根據距離找出對應的 station ID index
station_distances = station_info_df['STA_DIST'].values
station_id_index = np.searchsorted(station_distances, distance_to_standard)

# 建立模型輸入特徵（需依照模型訓練順序排列）
features = np.column_stack([
    [np.log(vs30)],
    [magnitude],
    [np.log(rupture_distance)],
    [fault_rake],
    [station_id_index]
])

# 使用模型預測 ln(PGA)
dmatrix = xgb.DMatrix(features)
predicted_lnPGA = model.predict(dmatrix)

print(f"Predicted ln(PGA): {predicted_lnPGA[0]:.4f}")
