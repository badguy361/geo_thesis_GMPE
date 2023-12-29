import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt

name = 'true'

#! 1. get result csv
def merge_scenario_result(name):
    df_site = pd.read_csv(f"scenario_result/{name}/sitemesh.csv", skiprows=[0])
    df_gmf = pd.read_csv(f"scenario_result/{name}/gmf-data.csv", skiprows=[0])
    df_total = df_gmf.merge(df_site, how='left', on='site_id')
    df_total = df_total.groupby("site_id").median()
    df_total.to_csv(f"scenario_result/{name}/chichi_scenario_record_{name}.csv",index=False)

#! 2. run Surfer to get grd file
# follow by scenario setting.png

# grd to xyz
def grd_to_csv(name):
    grd2xyz = pygmt.grd2xyz(grid=f'scenario_result/{name}/chichi_interpolate_{name}.grd',output_type='pandas')
    grd2xyz.to_csv(f'scenario_result/{name}/chichi_interpolate_{name}.csv',header=['STA_Lon_X','STA_Lat_Y','PGA'],index=False)
