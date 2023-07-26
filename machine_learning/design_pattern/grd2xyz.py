import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt

name = 'lin2009'
# grd to xyz
grf2xyz = pygmt.grd2xyz(grid=f'chichi_{name}.grd',output_type='pandas')
grf2xyz.to_csv(f'chichi_{name}.csv',header=['STA_Lon_X','STA_Lat_Y','PGA'],index=False)
