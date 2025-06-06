import sys
sys.path.append("..")
sys.path.append("../modules/gsim")
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
from modules.gsim.utils.imt import PGA, SA, PGV
from modules.gsim.phung_2020 import PhungEtAl2020Asc
from modules.gsim.chang_2023 import Chang2023
from modules.gsim.lin_2009 import Lin2009
from modules.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from modules.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from modules.gsim.chao_2020 import ChaoEtAl2020Asc

class SeismicDataPlotter:
    def __init__(self, data_list):
        """
        初始化資料
        :param data_list: list，包含 [x_train, x_test, y_train, y_test]
                          其中 x_train 與 x_test 是 np.ndarray，shape 為 (N, 5)
                          ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
        """
        if len(data_list) != 4:
            raise ValueError("資料輸入應為長度為 4 的 list：[x_train, x_test, y_train, y_test]")

        self.x_train, self.x_test, self.y_train, self.y_test = data_list

    def plot_mw_distribution(self):
        """
        繪製 MW 對應的事件數量圖：
        - bin 從 3.5 到 7.5，每 0.1 一格
        - x 軸顯示 3.5, 4.0, ..., 7.0
        - y 軸為線性比例
        - train 資料用藍色，test 資料用橘色
        """
        bins = np.arange(3.5, 8.0 + 0.1, 0.1)

        mw_train = self.x_train[:, 1]
        mw_test = self.x_test[:, 1]

        plt.figure(figsize=(12, 6))

        # 繪製 Train 資料直方圖
        plt.hist(
            mw_train,
            bins=bins,
            color='steelblue',
            edgecolor='black',
            alpha=0.7,
            label='Train'
        )

        # 疊加繪製 Test 資料直方圖
        plt.hist(
            mw_test,
            bins=bins,
            color='orange',
            edgecolor='black',
            alpha=0.7,
            label='Test'
        )

        plt.xlabel('Mw', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Earthquake Event Count by Mw (Train vs Test)', fontsize=14)

        # 設定 x 軸的 ticks
        plt.xticks(np.arange(3.5, 8.1, 0.5))
        plt.yticks([1, 10, 100, 1000], labels=["$10^0$", "$10^1$", "$10^2$", "$10^3$"])
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_fault_type_distribution(self):
        """
        使用 plt.bar 畫出 fault.type 的事件數分布圖（bar 寬度適中，不擠在一起）：
        - x 軸為 fault.type 類別
        - y 軸為每個類別的事件數
        - Train 與 Test 用不同顏色，side-by-side 顯示
        """

        fault_train = self.x_train[:, 3]
        fault_test = self.x_test[:, 3]

        # 獲取所有類別，排序後建立對應位置
        all_types = sorted(set(fault_train) | set(fault_test))
        x = np.arange(len(all_types))

        # 計數
        train_counts = collections.Counter(fault_train)
        test_counts = collections.Counter(fault_test)

        train_values = [train_counts.get(t, 0) for t in all_types]
        test_values = [test_counts.get(t, 0) for t in all_types]

        width = 0.35  # 每個 bar 寬度（較窄）

        plt.figure(figsize=(10, 6))
        plt.bar(x, train_values, width=width, color='steelblue', label='Train', edgecolor='black')
        plt.bar(x, test_values, width=width, color='orange', label='Test', edgecolor='black')

        plt.xticks(x, all_types)
        plt.xlabel('Fault Type', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Event Count by Fault Type (Train vs Test)', fontsize=14)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def plot_ln_pga_distribution(self):
        """
        繪製 ln(PGA)（即 y_train/y_test）的直方圖分布：
        - x 軸從 -2.0 到 1.0，每 0.1 一格
        - y 軸為對數刻度（10^0, 10^1, ..., 10^4）
        - Train 為藍色，Test 為橘色
        """
        bins = np.arange(-0.5, 7 + 0.1, 0.5)

        plt.figure(figsize=(12, 6))

        plt.hist(
            self.y_train,
            bins=bins,
            color='steelblue',
            edgecolor='black',
            alpha=0.7,
            label='Train'
        )

        plt.hist(
            self.y_test,
            bins=bins,
            color='orange',
            edgecolor='black',
            alpha=0.7,
            label='Test'
        )

        plt.yscale('log')
        plt.yticks([10**i for i in range(0, 5)], [f'$10^{i}$' for i in range(0, 5)])

        plt.xticks(np.arange(-0.5, 7.1, 0.5))
        plt.xlabel('ln(PGA) (gal)', fontsize=12)
        plt.ylabel('Number of Events', fontsize=12)
        plt.title('Distribution of ln(PGA) (Train vs Test)', fontsize=14)
        plt.legend()
        plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_event_distribution_map(self, eq_distribution_file: pd.DataFrame):
        """
        繪製地震事件分布圖（依 MW 調整圓圈大小）並顯示台灣大致輪廓：
        - 使用 PyGMT 繪製地圖
        - 不需本地 shapefile
        """
        import pygmt
        import pandas as pd
        import numpy as np

        # 去除重複 EQ_ID
        df = eq_distribution_file.drop_duplicates(subset='EQ_ID')

        # 檢查必要欄位
        required_columns = {'Hyp.Lat', 'Hyp.Long', 'MW'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # 取得地震資訊
        lats = df['Hyp.Lat'].values
        longs = df['Hyp.Long'].values
        mags = df['MW'].values
        sizes = (mags ** 3) * 0.002  # 調整圓圈大小比例以適應 PyGMT

        # 創建 PyGMT 圖
        fig = pygmt.Figure()

        # 設定台灣區域範圍
        region = [119.5, 122.5, 21.5, 25.5]

        # 繪製海岸線地圖
        fig.coast(
            region=region,
            projection="M10c",  # 墨卡托投影，10cm 寬度
            land="lightgray",
            water="white",
            shorelines="0.5p,black",
            frame=["ag", "+tEarthquake Distribution in Taiwan"],  # 包含標題
        )

        # 繪製地震事件
        fig.plot(
            x=longs,
            y=lats,
            size=sizes,
            style="cc",  # 圓形標記（cc 表示以厘米為單位的圓）
            fill="#DC143C",
            transparency=50,  # 透明度 50%
            pen="0.5p,black",  # 黑色邊框
        )

        # 顯示圖表
        fig.show()
    
    def get_data_less_than_10km(self, total_data: pd.DataFrame):
        """
        取得距離小於10km的資料, 儲存成csv檔案

        Args:
            total_data ([array]): [original dataset sperate by Mw]

        Returns:
            data_less_than_10km ([array]): [data of less than 10km]
        """
        data_less_than_10km = total_data[total_data['Rrup'] < 10]
        data_less_than_10km.to_csv('../../../data_less_than_10km.csv', index=False)

    def predict_less_than_10km_data(self, data_less_than_10km):
        """
        計算距離縮放圖

        Args:
            total_Mw_data ([array]): [original dataset sperate by Mw]
            model_path ([str]): [the place which the model be stored]
        
        Returns:
            各GMPE結果([array]): [residual of less than 10km]
        """
        dtype = [('vs30', '<f8'), ('mag', '<f8'), ('rrup', '<f8'),
                ('rake', '<f8'), ('sta_id', '<i8')]
        ctx = np.empty(len(data_less_than_10km), dtype=dtype)
        index = 0
        for record in range(len(data_less_than_10km)):  # 依照station_num、Rrup的順序建recarray
            ctx[index] = (data_less_than_10km.iloc[record]["Vs30"], data_less_than_10km.iloc[record]["MW"], data_less_than_10km.iloc[record]["Rrup"],
                        data_less_than_10km.iloc[record]["Rake"], data_less_than_10km.iloc[record]["STA_ID_int"])
            index += 1
        ctx = ctx.view(np.recarray)

        imts = [PGA()]
        ch_mean = [[0] * len(imts)]
        ch_sig = [[0] * len(imts)]
        ch_tau = [[0] * len(imts)]
        ch_phi = [[0] * len(imts)]

        # calculate Chang2023 total station value
        chang = Chang2023("model/no SMOGN/XGB_PGA.json")
        ch_mean, ch_sig, ch_tau, ch_phi = chang.compute(
            ctx, imts, ch_mean, ch_sig, ch_tau, ch_phi)
        ch_mean = np.exp(ch_mean)
        print(ch_mean[0])
        data_less_than_10km["Chang2023"] = ch_mean[0]

        # others GMM
        dtype = [('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
                ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
                ('hypo_depth', '<f8'), ('z2pt5', '<f8')]
        ctx = np.empty(len(data_less_than_10km), dtype=dtype)
        index = 0
        for record in range(len(data_less_than_10km)):
            ctx[index] = (40, data_less_than_10km.iloc[record]["MW"], data_less_than_10km.iloc[record]["Rake"], 0, data_less_than_10km.iloc[record]["Vs30"],
                        1, data_less_than_10km.iloc[record]["Rrup"], data_less_than_10km.iloc[record]["Rrup"], data_less_than_10km.iloc[record]["Rrup"], data_less_than_10km.iloc[record]["Rrup"], 10, True, 10, 1)
            index += 1
        ctx = ctx.view(np.recarray)

        imts = [PGA()]
        phung = PhungEtAl2020Asc()
        ph_mean = [[0] * len(imts)]
        ph_sig = [[0] * len(imts)]
        ph_tau = [[0] * len(imts)]
        ph_phi = [[0] * len(imts)]
        ph_mean, ph_sig, ph_tau, ph_phi = phung.compute(
            ctx, imts, ph_mean, ph_sig, ph_tau, ph_phi)
        ph_mean = np.exp(ph_mean)
        data_less_than_10km["Phung2020"] = ph_mean[0]
        lin = Lin2009()
        lin_mean = [[0] * len(imts)]
        lin_sig = [[0] * len(imts)]
        lin_tau = [[0] * len(imts)]
        lin_phi = [[0] * len(imts)]
        lin_mean, lin_sig = lin.compute(
            ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
        lin_mean = np.exp(lin_mean)
        data_less_than_10km["Lin2009"] = lin_mean[0]
        abrahamson = AbrahamsonEtAl2014()
        abr_mean = [[0] * len(imts)]
        abr_sig = [[0] * len(imts)]
        abr_tau = [[0] * len(imts)]
        abr_phi = [[0] * len(imts)]
        abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(
            ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
        abr_mean = np.exp(abr_mean)
        data_less_than_10km["Abrahamson2014"] = abr_mean[0]
        campbell = CampbellBozorgnia2014()
        cam_mean = [[0] * len(imts)]
        cam_sig = [[0] * len(imts)]
        cam_tau = [[0] * len(imts)]
        cam_phi = [[0] * len(imts)]
        cam_mean, cam_sig, cam_tau, cam_phi = campbell.compute(
            ctx, imts, cam_mean, cam_sig, cam_tau, cam_phi)
        cam_mean = np.exp(cam_mean)
        data_less_than_10km["Campbell2014"] = cam_mean[0]
        choa = ChaoEtAl2020Asc()
        choa_mean = [[0] * len(imts)]
        choa_sig = [[0] * len(imts)]
        choa_tau = [[0] * len(imts)]
        choa_phi = [[0] * len(imts)]
        choa_mean, choa_sig, choa_tau, choa_phi = choa.compute(
            ctx, imts, choa_mean, choa_sig, choa_tau, choa_phi)
        choa_mean = np.exp([choa_mean])
        data_less_than_10km["Chao2020"] = choa_mean[0]

        data_less_than_10km.to_csv('../../../data_less_than_10km_predicted.csv', index=False)

    def calculate_and_plot_residual_std(self, data_less_than_10km_predicted):
        """
        計算residual的標準差

        Args:
            data_less_than_10km_predicted ([array]): [residual of less than 10km]

        Returns:
            residual_std ([array]): [residual std of less than 10km]
        """
        data_less_than_10km_predicted["residual_Chang2023"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Chang2023"])
        data_less_than_10km_predicted["residual_Phung2020"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Phung2020"])
        data_less_than_10km_predicted["residual_Lin2009"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Lin2009"])
        data_less_than_10km_predicted["residual_Abrahamson2014"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Abrahamson2014"])
        data_less_than_10km_predicted["residual_Campbell2014"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Campbell2014"])
        data_less_than_10km_predicted["residual_Chao2020"] = np.log(data_less_than_10km_predicted["PGA"]) - np.log(data_less_than_10km_predicted["Chao2020"])

        print("Chang2023: ", data_less_than_10km_predicted["residual_Chang2023"].std())
        print("Phung2020: ", data_less_than_10km_predicted["residual_Phung2020"].std())
        print("Lin2009: ", data_less_than_10km_predicted["residual_Lin2009"].std())
        print("Abrahamson2014: ", data_less_than_10km_predicted["residual_Abrahamson2014"].std())
        print("Campbell2014: ", data_less_than_10km_predicted["residual_Campbell2014"].std())
        print("Chao2020: ", data_less_than_10km_predicted["residual_Chao2020"].std())
