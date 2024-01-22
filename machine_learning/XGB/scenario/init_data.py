import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt
from pykrige import OrdinaryKriging
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Path, PathPatch

name = 'Lin2009'
# df = pd.read_csv('../../../../TSMIP_FF.csv')
# chichi_df = df[df["MW"] == 7.65]  # choose chichi eq
# chichi_df.to_csv("chichi_ori.csv",index=False,columns=["STA_Lon_X","STA_Lat_Y","PGA"])

df_ori_true = pd.read_csv("scenario_result/true/chichi_ori.csv")
df_ori_chang = pd.read_csv(
    "scenario_result/Chang2023/chichi_scenario_record_Chang2023.csv")
df_ori_lin = pd.read_csv(
    "scenario_result/Lin2009/chichi_scenario_record_Lin2009.csv")
df_ori_dict = {
    "true": df_ori_true,
    "Lin2009": df_ori_lin,
    "Chang2023": df_ori_chang
}

#! 1. get result csv
def merge_scenario_result(name):
    df_site = pd.read_csv(f"scenario_result/{name}/dataset/sitemesh.csv", skiprows=[0])
    df_gmf = pd.read_csv(f"scenario_result/{name}/dataset/gmf-data.csv", skiprows=[0])
    df_total = df_gmf.merge(df_site, how='left', on='site_id')
    df_total = df_total.groupby("site_id").median()
    df_total.to_csv(f"scenario_result/{name}/chichi_scenario_record_{name}.csv",index=False)

#! 2. run Surfer to get grd file
def get_interpolation(name, df):
    PGA = np.array(df["PGA"])
    lons = np.array(df["STA_Lon_X"])
    lats = np.array(df["STA_Lat_Y"])
    # Kriging內插
    grid_space = 0.01
    grid_lon = np.arange(120, 122, grid_space)
    grid_lat = np.arange(21.6, 25.3, grid_space)
    OK = OrdinaryKriging(lons, lats, PGA, variogram_model='linear', variogram_parameters={
        'slope': 0.0101, 'nugget': 0}, verbose=True, enable_plotting=False, nlags=20)
    z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
    results = pd.DataFrame({
        'STA_Lon_X': xintrp.ravel(),
        'STA_Lat_Y': yintrp.ravel(),
        'PGA': z1.ravel()
    })
    results.to_csv(f'scenario_result/{name}/chichi_kriging_interpolate_{name}.csv', index=False)

_ = get_interpolation(name, df_ori_dict[name])