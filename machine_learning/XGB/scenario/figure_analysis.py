import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt
from scipy.interpolate import griddata
from pykrige import OrdinaryKriging

name = 'Chang2023'
# df = pd.read_csv('../../../../TSMIP_FF.csv')
# chichi_df = df[df["MW"] == 7.65]  # choose chichi eq
# chichi_df.to_csv("chichi_ori.csv",index=False,columns=["STA_Lon_X","STA_Lat_Y","PGA"])

df_ori_true =  pd.read_csv("scenario_result/true/chichi_ori.csv")
df_ori_chang = pd.read_csv("scenario_result/Chang2023/chichi_scenario_record_Chang2023.csv")
df_ori_lin = pd.read_csv("scenario_result/Lin2009/chichi_scenario_record_Lin2009.csv")
df_ori_dict = {
    "true": df_ori_true,
    "Lin2009": df_ori_lin,
    "Chang2023": df_ori_chang
}

df_chang = pd.read_csv(
    'scenario_result/Chang2023/chichi_interpolate_Chang2023.csv')
df_lin = pd.read_csv(f'scenario_result/Lin2009/chichi_interpolate_Lin2009.csv')
df_true = pd.read_csv(f'scenario_result/true/chichi_interpolate_true.csv')
df_dict = {
    "true": df_true,
    "Lin2009": df_lin,
    "Chang2023": df_chang
}

fault_data = [
    120.6436, 23.6404, 120.6480, 23.6424, 120.6511, 23.6459, 120.6543, 23.6493,
    120.6574, 23.6528, 120.6601, 23.6566, 120.6632, 23.6600, 120.6665, 23.6633,
    120.6707, 23.6656, 120.6752, 23.6675, 120.6796, 23.6694, 120.6838, 23.6716,
    120.6871, 23.6750, 120.6904, 23.6783, 120.6938, 23.6815, 120.6963, 23.6854,
    120.6980, 23.6896, 120.6996, 23.6939, 120.7013, 23.6981, 120.7023, 23.7025,
    120.7032, 23.7070, 120.7039, 23.7114, 120.7038, 23.7159, 120.7022, 23.7202,
    120.7005, 23.7244, 120.6987, 23.7286, 120.6987, 23.7331, 120.6984, 23.7376,
    120.6987, 23.7418, 120.7000, 23.7461, 120.7013, 23.7505, 120.7025, 23.7549,
    120.7055, 23.7584, 120.7090, 23.7616, 120.7095, 23.7654, 120.7081, 23.7697,
    120.7082, 23.7740, 120.7096, 23.7783, 120.7104, 23.7828, 120.7097, 23.7871,
    120.7086, 23.7914, 120.7110, 23.7952, 120.7126, 23.7994, 120.7119, 23.8029,
    120.7080, 23.8056, 120.7072, 23.8097, 120.7083, 23.8141, 120.7097, 23.8184,
    120.7097, 23.8228, 120.7090, 23.8273, 120.7094, 23.8318, 120.7078, 23.8359,
    120.7046, 23.8393, 120.7037, 23.8436, 120.7034, 23.8481, 120.7042, 23.8525,
    120.7053, 23.8569, 120.7062, 23.8613, 120.7057, 23.8658, 120.7068, 23.8702,
    120.7087, 23.8742, 120.7126, 23.8766, 120.7144, 23.8808, 120.7132, 23.8851,
    120.7121, 23.8895, 120.7106, 23.8938, 120.7082, 23.8977, 120.7060, 23.9016,
    120.7048, 23.9060, 120.7037, 23.9103, 120.7038, 23.9148, 120.7051, 23.9192,
    120.7065, 23.9235, 120.7077, 23.9279, 120.7075, 23.9324, 120.7045, 23.9359,
    120.7039, 23.9403, 120.7021, 23.9442, 120.6998, 23.9480, 120.6958, 23.9504,
    120.6926, 23.9537, 120.6891, 23.9567, 120.6866, 23.9605, 120.6875, 23.9648,
    120.6900, 23.9687, 120.6921, 23.9725, 120.6909, 23.9769, 120.6900, 23.9813,
    120.6898, 23.9858, 120.6896, 23.9904, 120.6895, 23.9949, 120.6905, 23.9992,
    120.6931, 24.0030, 120.6961, 24.0066, 120.6985, 24.0105, 120.6987, 24.0150,
    120.6982, 24.0195, 120.6982, 24.0239, 120.6994, 24.0283, 120.6978, 24.0324,
    120.6952, 24.0362, 120.6923, 24.0398, 120.6914, 24.0441, 120.6929, 24.0484,
    120.6963, 24.0514, 120.6999, 24.0542, 120.6999, 24.0587, 120.7003, 24.0631,
    120.7022, 24.0673, 120.7053, 24.0702, 120.7101, 24.0710, 120.7149, 24.0718,
    120.7187, 24.0746, 120.7218, 24.0779, 120.7232, 24.0821, 120.7258, 24.0856,
    120.7306, 24.0863, 120.7355, 24.0870, 120.7357, 24.0905, 120.7326, 24.0940,
    120.7296, 24.0975, 120.7305, 24.1019, 120.7315, 24.1064, 120.7324, 24.1108,
    120.7338, 24.1151, 120.7349, 24.1195, 120.7353, 24.1240, 120.7353, 24.1285,
    120.7336, 24.1327, 120.7331, 24.1371, 120.7348, 24.1412, 120.7366, 24.1454,
    120.7384, 24.1496, 120.7385, 24.1540, 120.7378, 24.1584, 120.7362, 24.1627,
    120.7344, 24.1669, 120.7323, 24.1710, 120.7302, 24.1751, 120.7271, 24.1784,
    120.7236, 24.1812, 120.7224, 24.1855, 120.7222, 24.1900, 120.7232, 24.1944,
    120.7242, 24.1988, 120.7251, 24.2033, 120.7255, 24.2077, 120.7245, 24.2119,
    120.7250, 24.2164, 120.7273, 24.2202, 120.7290, 24.2242, 120.7293, 24.2288,
    120.7300, 24.2332, 120.7321, 24.2372, 120.7349, 24.2409, 120.7378, 24.2446,
    120.7394, 24.2486, 120.7421, 24.2524, 120.7444, 24.2564, 120.7466, 24.2604,
    120.7492, 24.2642, 120.7519, 24.2678, 120.7542, 24.2718, 120.7573, 24.2752,
    120.7595, 24.2792, 120.7634, 24.2818, 120.7662, 24.2851, 120.7705, 24.2835,
    120.7754, 24.2840, 120.7803, 24.2843, 120.7852, 24.2844, 120.7901, 24.2845,
    120.7950, 24.2847, 120.7996, 24.2864, 120.8042, 24.2881, 120.8084, 24.2903,
    120.8128, 24.2925, 120.8171, 24.2946, 120.8211, 24.2973, 120.8246, 24.3005,
    120.8276, 24.3040, 120.8299, 24.3080, 120.8310, 24.3123, 120.8323, 24.3167,
    120.8334, 24.3203
]

hyp_lat = 23.85
hyp_lon = 120.82


def hazard_distribution(name, df, interpolate, fault_data, hyp_lat, hyp_lon):
    # before interpolate
    PGA = df["PGA"]
    x = df["STA_Lon_X"]
    y = df["STA_Lat_Y"]

    fig = pygmt.Figure()
    region = [119.5, 122.5, 21.5, 25.5]
    fig.basemap(region=region,
                projection="M12c",
                frame=["af", f"WSne+tchichi earthquake Hazard Distribution"])
    fig.coast(land="gray", water="gray")
    pygmt.makecpt(cmap="turbo", series=(0, 1.5, 0.1))
    fig.plot(x=x, y=y, style="c0.2c", cmap=True, color=PGA)
    fig.coast(shorelines="1p,black")
    fig.plot(x=fault_data[::2], y=fault_data[1::2], pen="thick,red")
    fig.plot(x=hyp_lon, y=hyp_lat, style='kstar4/0.3c', color="red")
    fig.colorbar(frame=["x+lPGA(g)"])
    if (interpolate):
        fig.savefig(
            f'scenario_result/{name}/chichi earthquake Hazard Distribution interpolate {name}.png', dpi=300)
    else:
        fig.savefig(f'scenario_result/{name}/chichi earthquake Hazard Distribution {name}.png', dpi=300)
    fig.show()

_ = hazard_distribution(name, df_ori_dict[name], False, fault_data, hyp_lat, hyp_lon)

def residual_distribution(name, df_dict, fault_data, hyp_lat, hyp_lon):
    PGA_residual = df_dict[name]["PGA"] - df_dict["true"]["PGA"]
    lon = df_dict[name]["STA_Lon_X"]
    lat = df_dict[name]["STA_Lat_Y"]

    fig = pygmt.Figure()
    region = [119.5, 122.5, 21.5, 25.5]
    fig.basemap(region=region,
                projection="M12c",
                frame=["af", f"WSne+tchichi earthquake Hazard Residual {name}"])
    pygmt.makecpt(cmap="turbo", series=(-0.5, 0.5, 0.1))
    fig.coast(land="gray", water="gray")
    fig.plot(x=lon, y=lat, style="c0.2c", cmap=True, color=PGA_residual)
    fig.plot(x=hyp_lon, y=hyp_lat, style='kstar4/0.3c', color="red")
    fig.plot(x=fault_data[::2], y=fault_data[1::2], pen="thick,red")
    fig.coast(shorelines="1p,black")
    fig.colorbar(frame=["x+lPGA(g)"])
    fig.savefig(
        f'scenario_result/{name}/chichi earthquake Hazard Distribution interpolate residual {name}.png', dpi=300)
    fig.show()

    return PGA_residual

def residual_statistic(name, PGA_residual):
    total_num_residual = [0] * 10
    total_num_residual[0] = np.count_nonzero((PGA_residual >= -0.5)
                                             & (PGA_residual < -0.4))
    total_num_residual[1] = np.count_nonzero((PGA_residual >= -0.4)
                                             & (PGA_residual < -0.3))
    total_num_residual[2] = np.count_nonzero((PGA_residual >= -0.3)
                                             & (PGA_residual < -0.2))
    total_num_residual[3] = np.count_nonzero((PGA_residual >= -0.2)
                                             & (PGA_residual < -0.1))
    total_num_residual[4] = np.count_nonzero((PGA_residual >= -0.1)
                                             & (PGA_residual < 0))
    total_num_residual[5] = np.count_nonzero((PGA_residual >= 0)
                                             & (PGA_residual < 0.1))
    total_num_residual[6] = np.count_nonzero((PGA_residual >= 0.1)
                                             & (PGA_residual < 0.2))
    total_num_residual[7] = np.count_nonzero((PGA_residual >= 0.2)
                                             & (PGA_residual < 0.3))
    total_num_residual[8] = np.count_nonzero((PGA_residual >= 0.3)
                                             & (PGA_residual < 0.4))
    total_num_residual[9] = np.count_nonzero((PGA_residual >= 0.4)
                                             & (PGA_residual <= 0.5))

    x_bar = np.linspace(-0.5, 0.5, 10)
    plt.bar(x_bar, total_num_residual, edgecolor='white', width=0.1, zorder=10)
    mu = np.mean(PGA_residual)
    sigma = np.std(PGA_residual)
    plt.text(-0.5, 1500, f'mean = {round(mu,2)}')
    plt.text(-0.5, 1000, f'sd = {round(sigma,2)}')
    plt.grid(linestyle=':', color='darkgrey', zorder=0)
    plt.xlabel('Total-Residual', fontsize=12)
    plt.ylabel('Numbers', fontsize=12)
    plt.title(f'{name} Total-Residual Distribution')
    plt.savefig(
        f'scenario_result/{name}/{name} Total-Residual Distribution.png',
        dpi=300)
    plt.show()

# PGA_residual = residual_distribution(
#     name, df_dict, fault_data, hyp_lat, hyp_lon)
# _ = residual_statistic(name, PGA_residual)
