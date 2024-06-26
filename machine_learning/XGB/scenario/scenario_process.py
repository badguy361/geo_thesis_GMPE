import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygmt
from pykrige import OrdinaryKriging
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Path, PathPatch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
# df = pd.read_csv('../../../../TSMIP_FF.csv')
# chichi_df = df[df["MW"] == 7.65]  # choose chichi eq
# chichi_df.to_csv("chichi_scenario_record_true.csv",index=False,columns=["STA_Lon_X","STA_Lat_Y","PGA"])


class scenario():
    def __init__(self, eq):
        self.eq = eq

    def merge_scenario_result(self, gmm):
        """
        To merge scenario result from the openquake scenario case

        Args:
            gmm ([str]): ground motion model which be used
        """
        df_site = pd.read_csv(
            f"{self.eq}/{gmm}/dataset/sitemesh.csv", skiprows=[0])
        df_gmf = pd.read_csv(
            f"{self.eq}/{gmm}/dataset/gmf-data.csv", skiprows=[0])
        df_total = df_gmf.merge(df_site, how='left', on='site_id')
        df_total = df_total.groupby("site_id").median()
        df_total = df_total.drop(columns=['event_id'])
        df_total.rename(columns={'lon': 'STA_Lon_X',
                        'lat': 'STA_Lat_Y', 'gmv_PGA': 'PGA'}, inplace=True)
        df_total.to_csv(
            f"{self.eq}/{gmm}/{self.eq}_scenario_record_{gmm}.csv", index=False)

    def get_interpolation(self, gmm, df):
        """
        To get interpolation file by Kriging

        Args:
            gmm ([str]): ground motion model which be used
            df ([dataframe]): dataframe which want to be interpolate
        """
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
        results.to_csv(
            f'{self.eq}/{gmm}/{self.eq}_kriging_interpolate_{gmm}.csv', index=False)

    def hazard_distribution(self, gmm, interpolate, fault_data, hyp_lat, hyp_lon):
        """
        To plot the hazard distribution in the area

        Args:
            gmm ([str]): ground motion model which be used
            interpolate ([bool]): if want to ues the interpolate dataset
            fault_data ([list]): fault which we focus
            hyp_lat ([str]): hypocenter lat
            hyp_lon ([str]): hypocenter lon
        """
        # max_color = 0.55
        # color_list = [round(i * 0.1, 1) for i in range(1,
        #                                                int(max_color * 10) + 1)]  # 0.1到max_color

        levels = [0.8, 2.5, 8.0, 25, 80, 140.0, 250.0, 440.0, 800.0]
        colors = ["#dffae0", "#39e838", "#fff62b", "#ff8a21", 
            "#ff5f2a", "#ce493c", "#a24d46", "#9e5586", "#c23bed"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, cmap.N)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        m = Basemap(llcrnrlon=119.9, llcrnrlat=21.6, urcrnrlon=122.2, urcrnrlat=25.4,
                    projection='merc', resolution='h', area_thresh=1000., ax=ax)
        m.drawcoastlines()
        plt.title(f'{self.eq} earthquake Hazard Distribution {gmm}')
        # 畫斷層
        # x_fault_map, y_fault_map = m(fault_data[::2], fault_data[1::2])
        hyp_lon_map, hyp_lat_map = m(hyp_lon, hyp_lat)
        # m.plot(x_fault_map, y_fault_map, 'r-', linewidth=2, zorder=10)
        m.scatter(hyp_lon_map, hyp_lat_map,
                  marker='*', color='r', zorder=10)
        # xy軸
        parallels = np.arange(21.5, 26.0, 0.5)
        m.drawparallels(parallels, labels=[
                        1, 0, 0, 0], fontsize=14, linewidth=0.0)
        meridians = np.arange(119.5, 122.5, 0.5)
        m.drawmeridians(meridians, labels=[
                        0, 0, 0, 1], fontsize=14, linewidth=0.0)

        # 繪製遮罩
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        map_edges = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])  # 地圖邊緣
        polys = [p.boundary for p in m.landpolygons]  # 陸地多邊形
        polys = [map_edges]+polys[:]  # 合併地圖邊緣和陸地多邊形
        codes = [
            [Path.MOVETO]+[Path.LINETO for p in p[1:]]
            for p in polys
        ]  # 定議遮罩繪製路徑
        polys_lin = [v for p in polys for v in p]
        codes_lin = [xx for cs in codes for xx in cs]
        path = Path(polys_lin, codes_lin)
        patch = PathPatch(path, facecolor='white', lw=0)  # 非陸地點畫白色
        ax.add_patch(patch)

        if (interpolate):
            df_int = pd.read_csv(
                f'{self.eq}/{gmm}/{self.eq}_kriging_interpolate_{gmm}.csv')
            PGA = np.array(df_int["PGA"])
            pga_gal = PGA*980
            lons = np.array(df_int["STA_Lon_X"])
            lats = np.array(df_int["STA_Lat_Y"])

            # 內插作圖
            x_map, y_map = m(lons, lats)

            # 連續g分布
            # scatter = m.scatter(x_map, y_map, c=PGA,
            #                     cmap='turbo', marker='o', edgecolor='none', vmin=0, vmax=max_color, s=4)
            # cbar = m.colorbar(scatter, boundaries=np.linspace(
            #     0, max_color, 15), location='right', pad="3%", extend='both', ticks=color_list)
            # cbar.set_label('PGA(g)')
            
            # 震級gal分布
            scatter = m.scatter(x_map, y_map, c=pga_gal,
                                cmap=cmap, norm=norm, marker='o', edgecolor='none', vmin=levels[0], vmax=levels[-1], s=4)
            cbar = m.colorbar(scatter, location='right', pad="3%", extend='both', ticks=levels)
            cbar.set_label('PGA(gal)')
            cbar.ax.text(0.5, 0.11, '1', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.22, '2', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.33, '3', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.44, '4', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.55, '5-', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.66, '5+', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.77, '6-', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.88, '6+', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)
            cbar.ax.text(0.5, 0.96, '7', transform=cbar.ax.transAxes, ha='center', va='center', color='black', fontsize=12)

            # 畫測站
            df_ori_true = pd.read_csv(
                f"{eq}/true/{eq}_scenario_record_true.csv")
            x_ori_true_map, y_ori_true_map = m(
                df_ori_true["STA_Lon_X"], df_ori_true["STA_Lat_Y"])
            m.scatter(x_ori_true_map, y_ori_true_map,c=df_ori_true["PGA"]*980,
                      cmap=cmap, norm=norm, marker='o', edgecolor='black', s=20)
            fig.savefig(
                f'{self.eq}/{gmm}/{self.eq} earthquake Hazard Distribution interpolate {gmm}.png', dpi=300)
        else:
            df = pd.read_csv(
                f'{self.eq}/{gmm}/chichi_scenario_record_{gmm}.csv')
            PGA = np.array(df["PGA"])
            lons = np.array(df["STA_Lon_X"])
            lats = np.array(df["STA_Lat_Y"])

            # 未內插作圖
            x_map, y_map = m(lons, lats)
            scatter = m.scatter(x_map, y_map, c=pga_gal,
                                cmap=cmap, marker='o', edgecolor='none', vmin=levels[0], vmax=levels[-1], s=4)
            cbar = m.colorbar(scatter, boundaries=levels, location='right', pad="3%", extend='both', ticks=levels)
            cbar.set_label('PGA(gal)')
            fig.savefig(
                f'{self.eq}/{gmm}/{self.eq} earthquake Hazard Distribution {gmm}.png', dpi=300)
        plt.show()

    def residual_distribution(self, gmm, fault_data, hyp_lat, hyp_lon):
        """
        To plot the hazard distribution residual in the area

        Args:
            gmm ([str]): ground motion model which be used
            fault_data ([list]): fault which we focus
            hyp_lat ([str]): hypocenter lat
            hyp_lon ([str]): hypocenter lon
        """
        df_int_true = pd.read_csv(
            f'{self.eq}/true/{self.eq}_kriging_interpolate_true.csv')
        df_int = pd.read_csv(
            f'{self.eq}/{gmm}/{self.eq}_kriging_interpolate_{gmm}.csv')

        PGA_residual = np.array(df_int["PGA"]) - np.array(df_int_true["PGA"])
        lons = df_int["STA_Lon_X"]
        lats = df_int["STA_Lat_Y"]

        # 作圖
        fig, ax = plt.subplots(figsize=(10, 10))
        m = Basemap(llcrnrlon=119.9, llcrnrlat=21.6, urcrnrlon=122.2, urcrnrlat=25.4,
                    projection='merc', resolution='h', area_thresh=1000., ax=ax)
        m.drawcoastlines()
        # 畫斷層
        x_fault_map, y_fault_map = m(fault_data[::2], fault_data[1::2])
        hyp_lon_map, hyp_lat_map = m(hyp_lon, hyp_lat)
        m.plot(x_fault_map, y_fault_map, 'r-', linewidth=2, zorder=10)
        m.scatter(hyp_lon_map, hyp_lat_map,
                  marker='*', color='r', zorder=10)
        # xy軸
        parallels = np.arange(21.5, 26.0, 0.5)
        m.drawparallels(parallels, labels=[
                        1, 0, 0, 0], fontsize=14, linewidth=0.0)
        meridians = np.arange(119.5, 122.5, 0.5)
        m.drawmeridians(meridians, labels=[
                        0, 0, 0, 1], fontsize=14, linewidth=0.0)

        # 繪製遮罩
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        map_edges = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])  # 地圖邊緣
        polys = [p.boundary for p in m.landpolygons]  # 陸地多邊形
        polys = [map_edges]+polys[:]  # 合併地圖邊緣和陸地多邊形
        codes = [
            [Path.MOVETO]+[Path.LINETO for p in p[1:]]
            for p in polys
        ]  # 定議遮罩繪製路徑
        polys_lin = [v for p in polys for v in p]
        codes_lin = [xx for cs in codes for xx in cs]
        path = Path(polys_lin, codes_lin)
        patch = PathPatch(path, facecolor='white', lw=0)  # 非陸地點畫白色
        ax.add_patch(patch)

        # 內插作圖
        colors = ["#0000a3", "#2525ff", "#fff5f5", "#fff5f5", "#ff3535", "#ea0000"]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=np.linspace(-0.3, 0.3, 7), ncolors=len(colors))

        x_map, y_map = m(lons, lats)
        scatter = m.scatter(x_map, y_map, c=PGA_residual,
                            cmap=cmap, norm=norm, marker='o', edgecolor='none', s=4)
        cbar = m.colorbar(scatter, boundaries=np.linspace(
            -0.3, 0.3, 7), location='right', pad="3%", extend='both', ticks=[-0.2, -0.1, 0.0, 0.1, 0.2])
        cbar.set_label('PGA(g)')

        # 畫測站
        df_ori_true = pd.read_csv(
            f"{self.eq}/true/{self.eq}_scenario_record_true.csv")
        x_ori_true_map, y_ori_true_map = m(
            df_ori_true["STA_Lon_X"], df_ori_true["STA_Lat_Y"])
        m.scatter(x_ori_true_map, y_ori_true_map,
                  marker='*', color="black", s=2)
        plt.title(f'{self.eq} earthquake Hazard Residual {gmm}')
        fig.savefig(
            f'{self.eq}/{gmm}/{self.eq} earthquake Hazard Distribution interpolate residual {gmm}.png', dpi=300)
        plt.show()

        return PGA_residual

    def residual_statistic(self, gmm, PGA_residual):
        """
        To statistic the hazard distribution residual 

        Args:
            gmm ([str]): ground motion model which be used
            PGA_residual ([list]): the residual from residual_distribution
        """
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
        plt.bar(x_bar, total_num_residual,
                edgecolor='white', width=0.1, zorder=10)
        mu = np.mean(PGA_residual)
        sigma = np.std(PGA_residual)
        plt.text(-0.5, 15000, f'mean = {round(mu,2)}')
        plt.text(-0.5, 10000, f'sd = {round(sigma,2)}')
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.xlabel('Total-Residual', fontsize=12)
        plt.ylabel('Numbers', fontsize=12)
        plt.title(f'{gmm} Total-Residual Distribution')
        plt.savefig(
            f'{self.eq}/{gmm}/{gmm} Total-Residual Distribution.png',
            dpi=300)
        plt.show()


if __name__ == '__main__':
    eq = "chichi"
    scenario_record_dict = {}
    cal_gmm = ['chang2023']
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
    hyp_lat = 23.85  # chichi: 23.85、0403: 23.77
    hyp_lon = 120.82  # chichi: 120.82、0403: 121.66

    sce = scenario(eq)
    for gmm in cal_gmm:
        if gmm != "true":
            print("merge ", gmm, " result")
            # _ = sce.merge_scenario_result(gmm)
        # scenario_record = pd.read_csv(
        #     f"{eq}/{gmm}/{eq}_scenario_record_{gmm}.csv")
        # _ = sce.get_interpolation(gmm, scenario_record)
        # _ = sce.hazard_distribution(
        #     gmm, True, fault_data, hyp_lat, hyp_lon)
        PGA_residual = sce.residual_distribution(
            gmm, fault_data, hyp_lat, hyp_lon)
        # _ = sce.residual_statistic(gmm, PGA_residual)
