import sys
sys.path.append("..")
sys.path.append("../modules/gsim")
from numpy.lib import recfunctions
from modules.gsim.utils.imt import PGA, SA, PGV
from modules.gsim.phung_2020 import PhungEtAl2020Asc
from modules.gsim.chang_2023 import Chang2023
from modules.gsim.lin_2009 import Lin2009
from modules.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from modules.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from modules.gsim.chao_2020 import ChaoEtAl2020Asc
from tqdm import tqdm
import xgboost as xgb
from modules.process_train import dataprocess
import pandas as pd
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap


class plot_fig:
    """

    Class method for plotting figure.

    """

    def __init__(self, model_name, abbreviation_name, dataset_type, target,
                x_total, y_total, predict_value, ori_full_data, score,
                Vs30, Mw, rrup, rake, station_id, station_num, station_map):
        """
        Args:
            x_total ([series]): [dataset feature test subset or all dataset]
            y_total ([series]): [dataset answer test subset or all dataset]
            predict_value ([series]): [predicted answer]
            ori_full_data ([series]): [ori_full_data]
            score ([float]): [R2 score from model]
        """
        self.model_name = model_name
        self.abbreviation_name = abbreviation_name
        self.dataset_type = dataset_type
        self.target = target
        self.fault_type_dict = {90: "REV", -90: "NM", 0: "SS"}
        self.period_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17,
                            0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
                            7.5, 10.0]
        
        self.Vs30 = Vs30
        self.Mw = Mw
        self.rrup = rrup
        self.rake = rake
        self.station_id = station_id
        self.station_num = station_num
        self.station_map = station_map

        self.x_total = x_total
        self.y_total = y_total
        self.predict_value = predict_value
        self.ori_full_data = ori_full_data
        self.score = score

    def data_distribution(self):
        """

        Plot the train & test data distribution.

        """
        unique_color = set(self.x_total[:, 3])
        color_map = {-90.0: ["r", "NM"], 0.0: ["g", "SS"], 90.0: ["b", "REV"]}

        # # Vs30 Mw relationship

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(np.exp(self.x_total[:, 0]),
                    self.x_total[:, 1],
                    c="gray",
                    marker='o',
                    facecolors='none',
                    s=8,
                    zorder=10)
        plt.xlabel('Vs30(m/s)', fontsize=12)
        plt.ylabel(f'Moment Magnitude, Mw', fontsize=12)
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/Vs30 Mw-dataset-distribution.jpg',
            dpi=300)
        plt.show()

        # Depth Mw relationship

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        for color in unique_color:
            indices = self.x_total[:, 3] == color
            plt.scatter(self.x_total[indices, -1], self.x_total[indices, 1],
                        facecolors='none', c=color_map[color][0],
                        s=8, zorder=10, label=color_map[color][1])
        plt.xlabel('Hypocenter Depth(km)', fontsize=12)
        plt.ylabel(f'Moment Magnitude, Mw', fontsize=12)
        plt.legend(loc="lower left")
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/Depth Mw-dataset-distribution.jpg',
            dpi=300)
        plt.show()

        # Rrup Mw relationship
        plt.grid(which="both",
                 axis="both",
                 linestyle="--",
                 linewidth=0.5,
                 alpha=0.5,
                 zorder=0)
        for color in unique_color:
            indices = self.x_total[:, 3] == color
            plt.scatter(np.exp(self.x_total[indices, 2]), self.x_total[indices, 1],
                        facecolors='none', c=color_map[color][0],
                        s=8, zorder=10, label=color_map[color][1])
        plt.legend(loc="lower left")
        plt.xlabel('Rupture Distance, Rrup(km)', fontsize=12)
        plt.ylabel(f'Moment Magnitude, Mw', fontsize=12)
        plt.xscale("log")
        plt.xlim(1 * 1e0, 7 * 1e2)
        plt.xticks([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500
        ], [
            1, '', '', '', '', '', '', '', '', 10, '', '', '', 50, '', '', '',
            '', 100, 200, 300, '', 500
        ])
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/Rrup-Mw dataset-distribution.jpg',
            dpi=300)
        plt.show()

        # fault type number relationship

        counter = Counter(self.x_total[:, 3])
        reverse = counter[90.0]
        normal = counter[-90.0]
        strike_slip = counter[0.0]

        plt.bar(["NN", "RE", "SS"], [normal, reverse, strike_slip],
                width=0.5, zorder=10)
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.xlabel('Fault type', fontsize=12)
        plt.ylabel(f'Numbers', fontsize=12)
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/Fault_type-dataset-distribution.jpg',
            dpi=300)
        plt.show()

    def residual(self):
        """

        Total residual and standard deviation and inter & intra event residual.
        Calculate residual by original anser and original predicted value.

        """

        # * 1. Vs30 Total Residual
        residual = self.predict_value - self.y_total
        residual_121 = []
        residual_199 = []
        residual_398 = []
        residual_794 = []
        residual_1000 = []

        for index, i in enumerate(np.exp(self.x_total[:, 0])):
            if i >= 121 and i < 199:
                residual_121.append(residual[index])
            elif i >= 199 and i < 398:
                residual_199.append(residual[index])
            elif i >= 398 and i < 794:
                residual_398.append(residual[index])
            elif i >= 794 and i < 1000:
                residual_794.append(residual[index])
            elif i >= 1000:
                residual_1000.append(residual[index])

        residual_121_mean = np.mean(residual_121)
        residual_199_mean = np.mean(residual_199)
        residual_398_mean = np.mean(residual_398)
        residual_794_mean = np.mean(residual_794)
        residual_1000_mean = np.mean(residual_1000)

        residual_121_std = np.std(residual_121)
        residual_199_std = np.std(residual_199)
        residual_398_std = np.std(residual_398)
        residual_794_std = np.std(residual_794)
        residual_1000_std = np.std(residual_1000)

        total_std = np.std(residual)
        print("total_std:",total_std)
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(np.exp(self.x_total[:, 0])[i], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)  # 看是落在哪個x網格
            y_net = (round(residual[i], 2) - (-3)) / (
                (3 - (-3)) / net)  # 看是落在哪個y網格
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格+=1
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(np.exp(self.x_total[:, 0])[j], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料網格的數值(zz值) 用以畫圖
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(np.exp(self.x_total[:, 0]),
                    residual,
                    s=8,
                    c=color_column,
                    cmap=newcmp,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.scatter([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean,
            residual_794_mean, residual_1000_mean
        ],
            marker='o',
            color='black',
            label='mean value',
            zorder=10)

        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean,
            residual_794_mean, residual_1000_mean
        ],
            'k--',
            label='1 std.',
            zorder=10)
        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean + residual_121_std, residual_199_mean +
            residual_199_std, residual_398_mean + residual_398_std,
            residual_794_mean + residual_794_std,
            residual_1000_mean + residual_1000_std
        ], 'k--',
            zorder=10)
        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean - residual_121_std, residual_199_mean -
            residual_199_std, residual_398_mean - residual_398_std,
            residual_794_mean - residual_794_std,
            residual_1000_mean - residual_1000_std
        ], 'k--',
            zorder=10)
        plt.xlim(0, 1400)
        plt.ylim(-3, 3)
        plt.xlabel('Vs30(m/s)', fontsize=12)
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)', fontsize=12)
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.2f std: %.2f'
            % (self.score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Vs30-{self.abbreviation_name} Predict Residual.png',
            dpi=300)
        plt.show()

        # * 2. Mw Toral Residual
        residual = self.predict_value - self.y_total
        residual_3_5 = []
        residual_4_5 = []
        residual_5_5 = []
        residual_6_5 = []

        for index, i in enumerate(self.x_total[:, 1]):
            if i >= 3.5 and i < 4.5:
                residual_3_5.append(residual[index])
            elif i >= 4.5 and i < 5.5:
                residual_4_5.append(residual[index])
            elif i >= 5.5 and i < 6.5:
                residual_5_5.append(residual[index])
            elif i >= 6.5:
                residual_6_5.append(residual[index])

        residual_3_5_mean = np.mean(residual_3_5)
        residual_4_5_mean = np.mean(residual_4_5)
        residual_5_5_mean = np.mean(residual_5_5)
        residual_6_5_mean = np.mean(residual_6_5)

        residual_3_5_std = np.std(residual_3_5)
        residual_4_5_std = np.std(residual_4_5)
        residual_5_5_std = np.std(residual_5_5)
        residual_6_5_std = np.std(residual_6_5)

        total_std = np.std(residual)

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(residual):
            x_net = (round(self.x_total[:, 1][i], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(self.x_total[:, 1][j], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(self.x_total[:, 1], residual,
                    s=8, c=color_column, cmap=newcmp,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.scatter([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean,
            residual_6_5_mean
        ],
            marker='o',
            color='black',
            label='mean value',
            zorder=10)

        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean,
            residual_6_5_mean
        ],
            'k--',
            label='1 std.',
            zorder=10)
        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean + residual_3_5_std, residual_4_5_mean +
            residual_4_5_std, residual_5_5_mean + residual_5_5_std,
            residual_6_5_mean + residual_6_5_std
        ], 'k--',
            zorder=10)
        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean - residual_3_5_std, residual_4_5_mean -
            residual_4_5_std, residual_5_5_mean - residual_5_5_std,
            residual_6_5_mean - residual_6_5_std
        ], 'k--',
            zorder=10)
        plt.xlim(3, 8)
        plt.ylim(-3, 3)
        plt.xlabel('Mw', fontsize=12)
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)', fontsize=12)
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.2f std: %.2f'
            % (self.score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Mw-{self.abbreviation_name} Predict Residual.png',
            dpi=300)
        plt.show()

        # *  3. Rrup Total Residual
        residual = self.predict_value - self.y_total
        residual_10 = []
        residual_31 = []
        residual_100 = []
        residual_316 = []

        for index, i in enumerate(np.exp(self.x_total[:, 2])):
            if i >= 10 and i < 31:
                residual_10.append(residual[index])
            elif i >= 31 and i < 100:
                residual_31.append(residual[index])
            elif i >= 100 and i < 316:
                residual_100.append(residual[index])
            elif i >= 316:
                residual_316.append(residual[index])

        residual_10_mean = np.mean(residual_10)
        residual_31_mean = np.mean(residual_31)
        residual_100_mean = np.mean(residual_100)
        residual_316_mean = np.mean(residual_316)

        residual_10_std = np.std(residual_10)
        residual_31_std = np.std(residual_31)
        residual_100_std = np.std(residual_100)
        residual_316_std = np.std(residual_316)

        total_std = np.std(residual)

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(residual):
            x_net = (round(np.exp(self.x_total[:, 2])[i], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(np.exp(self.x_total[:, 2])[j], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(which="both",
                 axis="both",
                 linestyle="--",
                 linewidth=0.5,
                 alpha=0.5,
                 zorder=0)
        plt.scatter(np.exp(self.x_total[:, 2]),
                    residual,
                    s=8,
                    c=color_column,
                    cmap=newcmp,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.scatter([10, 31, 100, 316], [
            residual_10_mean, residual_31_mean, residual_100_mean,
            residual_316_mean
        ],
            marker='o',
            color='black',
            label='mean value',
            zorder=10)

        plt.plot([10, 31, 100, 316], [
            residual_10_mean, residual_31_mean, residual_100_mean,
            residual_316_mean
        ],
            'k--',
            label='1 std.',
            zorder=10)
        plt.plot([10, 31, 100, 316], [
            residual_10_mean + residual_10_std, residual_31_mean +
            residual_31_std, residual_100_mean + residual_100_std,
            residual_316_mean + residual_316_std
        ], 'k--',
            zorder=10)
        plt.plot([10, 31, 100, 316], [
            residual_10_mean - residual_10_std, residual_31_mean -
            residual_31_std, residual_100_mean - residual_100_std,
            residual_316_mean - residual_316_std
        ], 'k--',
            zorder=10)
        plt.xscale("log")
        plt.xticks([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500
        ], [
            1, '', '', '', '', '', '', '', '', 10, '', '', '', 50, '', '', '',
            '', 100, 200, 300, '', 500
        ])
        plt.xlim(1 * 1e0, 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Rrup(km)', fontsize=12)
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)', fontsize=12)
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.2f std: %.2f'
            % (self.score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Rrup-{self.abbreviation_name} Predict Residual.png',
            dpi=300)
        plt.show()

        # * 4. Total Residual Number Distribution
        total_num_residual = [0] * 41
        total_num_residual[10] = np.count_nonzero((residual >= -2)
                                                  & (residual < -1.8))
        total_num_residual[11] = np.count_nonzero((residual >= -1.8)
                                                  & (residual < -1.6))
        total_num_residual[12] = np.count_nonzero((residual >= -1.6)
                                                  & (residual < -1.4))
        total_num_residual[13] = np.count_nonzero((residual >= -1.4)
                                                  & (residual < -1.2))
        total_num_residual[14] = np.count_nonzero((residual >= -1.2)
                                                  & (residual < -1.0))
        total_num_residual[15] = np.count_nonzero((residual >= -1.0)
                                                  & (residual < -0.8))
        total_num_residual[16] = np.count_nonzero((residual >= -0.8)
                                                  & (residual < -0.6))
        total_num_residual[17] = np.count_nonzero((residual >= -0.6)
                                                  & (residual < -0.4))
        total_num_residual[18] = np.count_nonzero((residual >= -0.4)
                                                  & (residual < -0.2))
        total_num_residual[19] = np.count_nonzero((residual >= -0.2)
                                                  & (residual < 0.0))
        total_num_residual[20] = np.count_nonzero((residual >= 0.0)
                                                  & (residual < 0.2))
        total_num_residual[21] = np.count_nonzero((residual >= 0.2)
                                                  & (residual < 0.4))
        total_num_residual[22] = np.count_nonzero((residual >= 0.4)
                                                  & (residual < 0.6))
        total_num_residual[23] = np.count_nonzero((residual >= 0.6)
                                                  & (residual < 0.8))
        total_num_residual[24] = np.count_nonzero((residual >= 0.8)
                                                  & (residual < 1.0))
        total_num_residual[25] = np.count_nonzero((residual >= 1.0)
                                                  & (residual < 1.2))
        total_num_residual[26] = np.count_nonzero((residual >= 1.2)
                                                  & (residual < 1.4))
        total_num_residual[27] = np.count_nonzero((residual >= 1.4)
                                                  & (residual < 1.6))
        total_num_residual[28] = np.count_nonzero((residual >= 1.6)
                                                  & (residual < 1.8))
        total_num_residual[29] = np.count_nonzero((residual >= 1.8)
                                                  & (residual < 2.0))
        total_num_residual[30] = np.count_nonzero((residual >= 2.0)
                                                  & (residual < 2.2))

        x_bar = np.linspace(-4, 4, 41)
        plt.bar(x_bar, total_num_residual, edgecolor='white', width=0.2,
                zorder=10)

        mu = np.mean(residual)
        sigma = np.std(residual)
        plt.text(2, 2700, f'mean = {round(mu,2)}', zorder=10)
        plt.text(2, 1700, f'sd = {round(sigma,2)}', zorder=10)
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.xlabel('Total-Residual', fontsize=12)
        plt.ylabel('Numbers', fontsize=12)
        plt.title(f'{self.abbreviation_name} Total-Residual Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} {self.abbreviation_name} Total-Residual Distribution.png',
            dpi=300)
        plt.show()
        """

        inter & intra event residual
        
        """

        # 計算inter-event(between-event) by mean value(共273顆地震)
        originaldata_predicted_result_df = pd.DataFrame(self.predict_value,
                                                        columns=['predicted'])
        total_data_df = pd.concat(
            [self.ori_full_data, originaldata_predicted_result_df], axis=1)
        # total_data_df["residual"] = np.abs((np.exp(total_data_df["predicted"]) - np.exp(total_data_df["lnPGA(gal)"]))/980)
        total_data_df["residual"] = total_data_df["predicted"] - total_data_df[
            f"ln{self.target}(gal)"]

        # build new dataframe to collect inter-event value
        summeries = {'residual': 'mean', 'MW': 'max'}
        inter_event = total_data_df.groupby(
            by="EQ_ID").agg(summeries).reset_index()
        inter_event = inter_event.rename(columns={
            'residual': 'inter_event_residual',
            'MW': 'Mw'
        })

        # * Mw inter-event
        Mw_yticks = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(inter_event['Mw'],
                    inter_event['inter_event_residual'],
                    marker='o',
                    s=8,
                    facecolors='None',
                    edgecolors='black',
                    zorder=10)
        plt.plot([3, 8], [
            inter_event['inter_event_residual'].mean(),
            inter_event['inter_event_residual'].mean()
        ],
            'b--',
            linewidth=0.5,
            label="mean value",
            zorder=10)
        plt.xlim(3, 8)
        plt.ylim(-1.6, 1.6)
        plt.yticks(Mw_yticks)
        plt.xlabel('Mw', fontsize=12)
        plt.ylabel(
            f'Inter-event Residual ln({self.target})(cm/s^2)', fontsize=12)
        inter_mw_mean = round(inter_event['inter_event_residual'].mean(), 2)
        inter_mw_std = round(inter_event['inter_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Inter-event Residual Mean:{inter_mw_mean} Std:{inter_mw_std}'
        )
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Mw-{self.abbreviation_name} Inter-event Residual.png',
            dpi=300)
        plt.show()

        # 計算intra-event(intra-event) by total residual - inter-event residual
        # merge inter-event dataframe to original dataframe
        total_data_df = pd.merge(total_data_df,
                                 inter_event,
                                 how='left',
                                 on=['EQ_ID'])
        total_data_df['intra_event_residual'] = total_data_df[
            'residual'] - total_data_df['inter_event_residual']

        # * Rrup intra-event
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(residual):
            x_net = (round(total_data_df['Rrup'][i], 2) - (-50)) / (
                (600 - (-50)) / net)  # -50為圖中最小值 600為圖中最大值
            y_net = (round(total_data_df['intra_event_residual'][i], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)  # -2.5為圖中最小值 2.5為圖中最大值
            if y_net <= net:
                zz[math.floor(x_net), math.floor(y_net)] += 1
            else:
                zz[math.floor(x_net), net-1] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(total_data_df['Rrup'][j], 2) -
                     (-50)) / ((600 - (-50)) / net)
            y_net = (round(total_data_df['intra_event_residual'][j], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            if y_net <= net:
                color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            else:
                color_column.append(zz[math.floor(x_net), net-1])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(which="both",
                 axis="both",
                 linestyle="--",
                 linewidth=0.5,
                 alpha=0.5,
                 zorder=0)
        plt.scatter(total_data_df['Rrup'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    c=color_column,
                    cmap=newcmp,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.plot([-50, 600], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
            'b--',
            label="mean value",
            linewidth=0.5,
            zorder=10)
        plt.xlim(1 * 1e0, 1e3)
        plt.ylim(-2.5, 2.5)
        plt.legend()
        plt.xscale("log")
        plt.xticks([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500
        ], [
            1, '', '', '', '', '', '', '', '', 10, '', '', '', 50, '', '', '',
            '', 100, 200, 300, '', 500
        ])
        plt.xlabel('Rrup(km)', fontsize=12)
        plt.ylabel(
            f'Intra-event Residual ln({self.target})(cm/s^2)', fontsize=12)
        intra_rrup_mean = round(
            total_data_df['intra_event_residual'].mean(), 2)
        intra_rrup_std = round(total_data_df['intra_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_rrup_mean} Std:{intra_rrup_std}'
        )
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Rrup-{self.abbreviation_name} Intra-event Residual.png',
            dpi=300)
        plt.show()

        # * Vs30 intra-event
        Vs30_xticks = [200, 400, 600, 800, 1000, 1200, 1400]
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(residual):
            x_net = (round(total_data_df['Rrup'][i], 2) - 0) / (
                (1400 - 0) / net)
            y_net = (round(total_data_df['intra_event_residual'][i], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            if x_net < net and y_net < net:
                zz[math.floor(x_net), math.floor(y_net)] += 1
            else:
                zz[net-1, net-1] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(total_data_df['Rrup'][j], 2) - 0) / (
                (1400 - 0) / net)
            y_net = (round(total_data_df['intra_event_residual'][j], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            if x_net < net and y_net < net:
                color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            else:
                color_column.append(zz[net-1, net-1])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(total_data_df['Vs30'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    c=color_column,
                    cmap=newcmp,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.plot([0, 1400], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
            'b--',
            label="mean value",
            linewidth=0.5,
            zorder=10)
        plt.xlim(0, 1400)
        plt.ylim(-2.5, 2.5)
        plt.xticks(Vs30_xticks)
        plt.legend()
        plt.xlabel('Vs30(m/s)', fontsize=12)
        plt.ylabel(
            f'Intra-event Residual ln({self.target})(cm/s^2)', fontsize=12)
        intra_vs30_mean = round(
            total_data_df['intra_event_residual'].mean(), 2)
        intra_vs30_std = round(total_data_df['intra_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_vs30_mean} Std:{intra_vs30_std}'
        )
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} Vs30-{self.abbreviation_name} Intra-event Residual.png',
            dpi=300)
        plt.show()

    def measured_predict(self, lowerbound, higherbound):
        """

        Plot the predict value and true value distribution .

        Args:
            lowerbound ([int]): [net lowerbound]
            higherbound ([int]): [net higherbound]
        """
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(self.y_total):
            x_net = (round(self.y_total[i], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            # +2:因為網格從-2開始打 10:頭減尾8-(-2) 10/net:網格間格距離 x_net:x方向第幾個網格
            y_net = (round(self.predict_value[i], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            if x_net < net and y_net < net:
                zz[math.floor(x_net), math.floor(y_net)] += 1
            else:
                zz[net-1, net-1] += 1
            i += 1

        j = 0
        while j < len(self.y_total):
            x_net = (round(self.y_total[j], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            y_net = (round(self.predict_value[j], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            if x_net < net and y_net < net:
                color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            else:
                color_column.append(zz[net-1, net-1])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', zorder=0)
        plt.scatter(self.y_total, self.predict_value, s=8,
                    c=color_column, cmap=newcmp, zorder=10)
        x_line = [lowerbound, higherbound]
        y_line = [lowerbound, higherbound]
        plt.plot(x_line, y_line, 'r--', alpha=0.5, zorder=10)
        plt.xlabel(f'Measured ln({self.target})(cm/s^2)', fontsize=12)
        plt.ylabel(f'Predict ln({self.target})(cm/s^2)', fontsize=12)
        plt.ylim(lowerbound, higherbound)
        plt.xlim(lowerbound, higherbound)
        plt.title(
            f'{self.dataset_type} {self.abbreviation_name} Measured Predicted Distribution'
        )
        plt.text(higherbound-4, lowerbound+2, f"R2 score = {round(self.score,2)}")
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.savefig(
            f'../{self.abbreviation_name}/{self.dataset_type} {self.target} {self.abbreviation_name} Measured Predicted Comparison.png',
            dpi=300)
        plt.show()

    def distance_scaling(self, avg_station_id, total_Mw_data, model_path):
        """

        Compute distance scaling figure follow condition given by ourself.

        Args:
            avg_station_id ([bool]): [to decide if we want to plot all station]
            total_data ([array]): [original dataset sperate by Mw]
            model_path ([str]): [the place which the model be stored]
        """

        # * comparsion ohter GMMs

        dtype = [('vs30', '<f8'), ('mag', '<f8'), ('rrup', '<f8'),
                ('rake', '<f8'), ('sta_id', '<i8')]
        rrup_num = [0.1, 0.5, 0.75, 1, 5, 10, 20, 30,
                    40, 50, 60, 70, 80, 90, 100, 150, 200, 300]
        total_elements = len(rrup_num) * self.station_num
        ctx = np.empty(total_elements, dtype=dtype)
        index = 0
        for station_id in range(self.station_num):  # 依照station_num、Rrup的順序建recarray
            Vs30 = self.station_map.iloc[station_id]["Vs30"]
            print(station_id,Vs30)
            for rrup in rrup_num:
                ctx[index] = (Vs30, self.Mw, rrup,
                            self.rake, station_id + 1)
                index += 1
        ctx = ctx.view(np.recarray)

        imts = [PGA()]
        ch_mean = [[0] * len(imts)]
        ch_sig = [[0] * len(imts)]
        ch_tau = [[0] * len(imts)]
        ch_phi = [[0] * len(imts)]

        # calculate Chang2023 total station value
        chang = Chang2023(model_path)
        ch_mean, ch_sig, ch_tau, ch_phi = chang.compute(
            ctx, imts, ch_mean, ch_sig, ch_tau, ch_phi)
        ch_mean = np.exp(ch_mean)
        split_mean = np.split(ch_mean[0], self.station_num) 
        # 預測結果依station_num數量分組 shape:(732,18)
        
        if avg_station_id:
            avg_mean = np.array(split_mean).mean(axis=0)
            plt.plot(rrup_num, avg_mean, 'r',
                    linewidth='1.6', label="This study avg", zorder=20)
        else:
            for i in range(self.station_num):
                plt.plot(rrup_num, split_mean[i],
                        'r', linewidth='0.4', zorder=5)

        # others GMM
        dtype = [('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
                ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
                ('hypo_depth', '<f8'), ('z2pt5', '<f8')]
        ctx = np.empty(len(rrup_num), dtype=dtype)
        index = 0
        for rrup in rrup_num:
            ctx[index] = (40, self.Mw, self.rake, 0, self.Vs30,
                          1, rrup, rrup, rrup, rrup, 10, True, 10, 1)
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
        lin = Lin2009()
        lin_mean = [[0] * len(imts)]
        lin_sig = [[0] * len(imts)]
        lin_tau = [[0] * len(imts)]
        lin_phi = [[0] * len(imts)]
        lin_mean, lin_sig = lin.compute(
            ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
        lin_mean = np.exp(lin_mean)
        abrahamson = AbrahamsonEtAl2014()
        abr_mean = [[0] * len(imts)]
        abr_sig = [[0] * len(imts)]
        abr_tau = [[0] * len(imts)]
        abr_phi = [[0] * len(imts)]
        abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(
            ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
        abr_mean = np.exp(abr_mean)
        campbell = CampbellBozorgnia2014()
        cam_mean = [[0] * len(imts)]
        cam_sig = [[0] * len(imts)]
        cam_tau = [[0] * len(imts)]
        cam_phi = [[0] * len(imts)]
        cam_mean, cam_sig, cam_tau, cam_phi = campbell.compute(
            ctx, imts, cam_mean, cam_sig, cam_tau, cam_phi)
        cam_mean = np.exp(cam_mean)
        choa = ChaoEtAl2020Asc()
        choa_mean = [[0] * len(imts)]
        choa_sig = [[0] * len(imts)]
        choa_tau = [[0] * len(imts)]
        choa_phi = [[0] * len(imts)]
        choa_mean, choa_sig, choa_tau, choa_phi = choa.compute(
            ctx, imts, choa_mean, choa_sig, choa_tau, choa_phi)
        choa_mean = np.exp([choa_mean])

        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.scatter(np.exp(total_Mw_data[0][0][:, 2]),
                    np.exp(total_Mw_data[0][1]) / 980,
                    marker='o',
                    facecolors='none',
                    s=2,
                    color='grey',
                    label='data',
                    zorder=20)
        plt.plot(rrup_num, ph_mean[0], 'orange',
                 linewidth='1', label="Phung2020", zorder=10)
        plt.plot(rrup_num, lin_mean[0], 'g',
                 linewidth='1', label="Lin2009", zorder=10)
        plt.plot(rrup_num, abr_mean[0], 'b',
                 linewidth='1', label="Abrahamson2014", zorder=10)
        plt.plot(rrup_num, cam_mean[0], 'yellow',
                 linewidth='1', label="CampbellBozorgnia2014", zorder=10)
        plt.plot(rrup_num, choa_mean[0], 'pink',
                 linewidth='1', label="ChaoEtAl2020Asc", zorder=10)
        plt.xlabel(f'Rrup(km)')
        plt.ylabel(f'{self.target}(g)')
        plt.title(
            f"Mw = {ctx['mag'][0]}, Fault = {self.fault_type_dict[ctx['rake'][0]]}")
        plt.ylim(10e-4, 5)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
                   [0.1, 0.5, 1, 10, 50, 100, 200, 300])
        plt.legend()
        plt.savefig(
            f"distance scaling Mw-{ctx['mag'][0]} Vs30-{ctx['vs30'][0]} fault-type-{self.fault_type_dict[ctx['rake'][0]]} avg_station_id-{avg_station_id}.jpg", dpi=300)
        plt.show()

        # * comparsion different Mw
        booster_PGA = xgb.Booster()
        booster_PGA.load_model(model_path)

        Mw_list = [4, 5, 6, 7]
        color = ["lightblue", "lightsalmon", "lightgreen", "lightcoral"]

        if avg_station_id:
            for i, Mw in enumerate(Mw_list):
                total_sta_predict = []
                for station_id in tqdm(range(self.station_num)):  # 預測所有station取平均
                    single_sta_predict = []
                    Vs30 = self.station_map.iloc[station_id]["Vs30"]
                    for rrup in rrup_num:
                        RSCon = xgb.DMatrix(
                            np.array([[np.log(Vs30), Mw,
                                       np.log(rrup), 90, station_id + 1]]))
                        single_sta_predict.append(
                            np.exp(booster_PGA.predict(RSCon)) / 980)
                    total_sta_predict.append(single_sta_predict)
                plt.scatter(np.exp(total_Mw_data[i+1][0][:, 2]),
                            np.exp(total_Mw_data[i+1][1]) / 980,
                            marker='o',
                            facecolors='none',
                            s=2,
                            c=color[i],
                            zorder=5)
                plt.plot(rrup_num, np.array(total_sta_predict).mean(axis=0),
                         linewidth='1.2', zorder=20, label=f"Mw:{Mw_list[i]}")

        else:
            for i, Mw in enumerate(Mw_list):
                single_sta_predict = []
                Vs30 = self.station_map.iloc[self.station_id-1]["Vs30"]
                for rrup in rrup_num:
                    RSCon = xgb.DMatrix(
                        np.array([[np.log(Vs30), Mw,
                                   np.log(rrup), self.rake, self.station_id]]))
                    single_sta_predict.append(
                        np.exp(booster_PGA.predict(RSCon)) / 980)
                plt.scatter(np.exp(total_Mw_data[i+1][0][:, 2]),
                            np.exp(total_Mw_data[i+1][1]) / 980,
                            marker='o',
                            facecolors='none',
                            s=2,
                            c=color[i],
                            zorder=5)
                plt.plot(rrup_num, single_sta_predict,
                         linewidth='1.2', zorder=20, label=f"Mw:{Mw_list[i]}")
                
        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.xlabel(f'Rrup(km)')
        plt.ylabel(f'{self.target}(g)')
        plt.title(
            f"Fault = {self.fault_type_dict[self.rake]}")
        plt.ylim(10e-4, 5)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
                   [0.1, 0.5, 1, 10, 50, 100, 200, 300])
        plt.legend(loc="lower left")
        plt.savefig(
            f"distance scaling {self.target} fault-type-{self.fault_type_dict[self.rake]} avg_station_id-{avg_station_id} station_id-{self.station_id}.jpg", dpi=300)
        plt.show()

    def explainable(self, eq_df, model_feture, ML_model, index_start, index_end):
        """

        This function shows explanation wihch include global explaination and local explaination of the model in the given target .

        Args:
            eq_df (eq dataset): [the eq dataset]
            model_feture ([list]): [input parameters]
            ML_model ([model]): [the model from sklearn or other package]
            index_start ([int]): [start record index]
            index_end ([int]): [end record index]
        """
        df = pd.DataFrame(self.x_total, columns=model_feture)
        explainer = shap.Explainer(ML_model)
        shap_values = explainer(df)

        #! Shap value(in map)
        # plt.scatter(eq_df['STA_Lon_X'][index_start], eq_df['STA_Lat_Y'][index_start], \
        #         s=8, c='black', zorder=20) # 測站位置
        # plt.text(eq_df['STA_Lon_X'][index_start], eq_df['STA_Lat_Y'][index_start], \
        #         f"{eq_df['STA_ID'][index_start]}_{eq_df['STA_rank'][index_start]}", \
        #         zorder=20)
        plt.scatter(eq_df["STA_Lon_X"],
                    eq_df["STA_Lat_Y"],
                    c=np.sum(shap_values.values[index_start:index_end], axis=1)
                    + shap_values.base_values[index_start],
                    cmap='cool',
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='SHAP value')
        cbar.set_label('SHAP value', fontsize=12)
        plt.xlim(119.4, 122.3)
        plt.ylim(21.8, 25.5)
        plt.xlabel('longitude', fontsize=12)
        plt.ylabel('latitude', fontsize=12)
        plt.title('TSMIP SHAP value in map')
        plt.savefig(f"SHAP value in map eq-{index_start} {self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        #! Station ID shap value
        fig = plt.figure()
        plt.scatter(eq_df["STA_Lon_X"],
                    eq_df["STA_Lat_Y"],
                    c=shap_values.values[index_start:index_end, 4]
                    + shap_values.base_values[index_start],
                    cmap='cool',
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='SHAP value')
        cbar.set_label('SHAP value', fontsize=12)
        plt.xlim(119.4, 122.3)
        plt.ylim(21.8, 25.5)
        plt.xlabel('longitude', fontsize=12)
        plt.ylabel('latitude', fontsize=12)
        plt.title('Station_ID SHAP value')
        plt.savefig(f"Station_ID eq-{index_start} {self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        #! Vs30 shap value
        fig = plt.figure()
        plt.scatter(eq_df["STA_Lon_X"],
                    eq_df["STA_Lat_Y"],
                    c=shap_values.values[index_start:index_end, 0]
                    + shap_values.base_values[index_start],
                    cmap='cool',
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='SHAP value')
        cbar.set_label('SHAP value', fontsize=12)
        plt.xlim(119.4, 122.3)
        plt.ylim(21.8, 25.5)
        plt.xlabel('longitude', fontsize=12)
        plt.ylabel('latitude', fontsize=12)
        plt.title('Vs30 SHAP value')
        plt.savefig(f"Vs30 eq-{index_start} {self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        #! Station ID and Vs30 residual shap value
        fig = plt.figure()
        plt.scatter(eq_df["STA_Lon_X"],
                    eq_df["STA_Lat_Y"],
                    c=shap_values.values[index_start:index_end, 0]
                    - shap_values.values[index_start:index_end, 4],
                    cmap='cool',
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='SHAP value')
        cbar.set_label('residual', fontsize=12)
        plt.xlim(119.4, 122.3)
        plt.ylim(21.8, 25.5)
        plt.xlabel('longitude', fontsize=12)
        plt.ylabel('latitude', fontsize=12)
        plt.title('TSMIP Station ID vs Vs30 SHAP value')
        plt.savefig(f"Station_ID and Vs30 eq-{index_start} {self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        #! Global Explainable
        # summary
        fig = plt.figure()
        shap.summary_plot(shap_values, df, show=False)
        plt.savefig(f"summary_plot_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # bar plot
        fig = plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_bar_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # scatter plot
        fig = plt.figure()
        shap.plots.scatter(shap_values[:, "MW"],
                           color=shap_values[:, "lnRrup"],
                           show=False)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_scatter_Mw_lnRrup_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        fig = plt.figure()
        shap.plots.scatter(shap_values[:, "lnRrup"],
                           color=shap_values[:, "MW"],
                           show=False)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_scatter_lnRrup_Mw_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # ! Local Explainable
        # waterfall
        fig = plt.figure()
        shap.plots.waterfall(shap_values[index_start],
                             show=False)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_waterfall_{index_start}_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # force plot
        shap.initjs()
        shap.force_plot(explainer.expected_value,
                        shap_values.values[index_start, :],
                        df.iloc[index_start, :],
                        show=False,
                        matplotlib=True)
        plt.savefig(f"force_plot_{index_start}_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        fig.canvas.draw()

    def respond_spectrum(self, plot_all_rake=False, avg_station_id=True, *args: "model"):
        """

        This function is called to plot respond spectrum .

        Args:
            plot_all_rake ([bool]): [to decide if plot all rake figure]
            avg_station_id ([bool]): [to decide if plot all station figure]
        """
        booster_PGA = xgb.Booster()
        booster_PGA.load_model(args[0])
        booster_PGV = xgb.Booster()
        booster_PGV.load_model(args[1])
        booster_Sa001 = xgb.Booster()
        booster_Sa001.load_model(args[2])
        booster_Sa002 = xgb.Booster()
        booster_Sa002.load_model(args[3])
        booster_Sa003 = xgb.Booster()
        booster_Sa003.load_model(args[4])
        booster_Sa004 = xgb.Booster()
        booster_Sa004.load_model(args[5])
        booster_Sa005 = xgb.Booster()
        booster_Sa005.load_model(args[6])
        booster_Sa0075 = xgb.Booster()
        booster_Sa0075.load_model(args[7])
        booster_Sa01 = xgb.Booster()
        booster_Sa01.load_model(args[8])
        booster_Sa012 = xgb.Booster()
        booster_Sa012.load_model(args[9])
        booster_Sa015 = xgb.Booster()
        booster_Sa015.load_model(args[10])
        booster_Sa017 = xgb.Booster()
        booster_Sa017.load_model(args[11])
        booster_Sa02 = xgb.Booster()
        booster_Sa02.load_model(args[12])
        booster_Sa025 = xgb.Booster()
        booster_Sa025.load_model(args[13])
        booster_Sa03 = xgb.Booster()
        booster_Sa03.load_model(args[14])
        booster_Sa04 = xgb.Booster()
        booster_Sa04.load_model(args[15])
        booster_Sa05 = xgb.Booster()
        booster_Sa05.load_model(args[16])
        booster_Sa075 = xgb.Booster()
        booster_Sa075.load_model(args[17])
        booster_Sa10 = xgb.Booster()
        booster_Sa10.load_model(args[18])
        booster_Sa15 = xgb.Booster()
        booster_Sa15.load_model(args[19])
        booster_Sa20 = xgb.Booster()
        booster_Sa20.load_model(args[20])
        booster_Sa30 = xgb.Booster()
        booster_Sa30.load_model(args[21])
        booster_Sa40 = xgb.Booster()
        booster_Sa40.load_model(args[22])
        booster_Sa50 = xgb.Booster()
        booster_Sa50.load_model(args[23])
        booster_Sa75 = xgb.Booster()
        booster_Sa75.load_model(args[24])
        booster_Sa100 = xgb.Booster()
        booster_Sa100.load_model(args[25])
        booster = [booster_Sa001, booster_Sa002, booster_Sa003, booster_Sa004,
                   booster_Sa005, booster_Sa0075, booster_Sa01, booster_Sa012, booster_Sa015, booster_Sa017,
                   booster_Sa02, booster_Sa025, booster_Sa03, booster_Sa04, booster_Sa05, booster_Sa075,
                   booster_Sa10, booster_Sa15, booster_Sa20, booster_Sa30, booster_Sa40, booster_Sa50, booster_Sa75,
                   booster_Sa100]

        # * 1. focal.type independent
        if plot_all_rake == True:
            for _rake in self.fault_type_dict:
                single_sta_predict = []
                RSCon = xgb.DMatrix(
                    np.array([[np.log(self.Vs30), self.Mw,
                               np.log(self.rrup), _rake, self.station_id]]))
                for _model in booster:
                    predict_value = np.exp(_model.predict(RSCon)) / 980
                    single_sta_predict.append(predict_value)
                plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                         single_sta_predict, label=self.fault_type_dict[_rake])

            plt.grid(which="both",
                     axis="both",
                     linestyle="-",
                     linewidth=0.5,
                     alpha=0.5)
            plt.title(f"Mw = {self.Mw}, Rrup = {self.rrup}km, Vs30 = {self.Vs30}m/s")
            plt.xlabel("Period(s)", fontsize=12)
            plt.ylabel("PSA(g)", fontsize=12)
            plt.ylim(10e-6, 1)
            plt.xlim(0.01, 10.0)
            plt.yscale("log")
            plt.xscale("log")
            plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                       [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
            plt.legend()
            plt.savefig(
                f"response spectrum-all_rake-{plot_all_rake} Mw-{self.Mw} Rrup-{self.rrup} Vs30-{self.Vs30} station-{self.station_id}.png",
                dpi=300)
            plt.show()

        else:
            single_sta_predict = []
            RSCon = xgb.DMatrix(
                np.array([[np.log(self.Vs30), self.Mw,
                           np.log(self.rrup), self.rake, self.station_id]]))
            for _model in booster:
                predict_value = np.exp(_model.predict(RSCon)) / 980
                single_sta_predict.append(predict_value)
            plt.grid(which="both",
                     axis="both",
                     linestyle="-",
                     linewidth=0.5,
                     alpha=0.5)
            plt.plot(self.period_list,
                     single_sta_predict, label=self.fault_type_dict[self.rake])
            plt.title(f"Mw = {self.Mw}, Rrup = {self.rrup}km, Vs30 = {self.Vs30}m/s")
            plt.xlabel("Period(s)", fontsize=12)
            plt.ylabel("PSA(g)", fontsize=12)
            plt.ylim(10e-6, 1)
            plt.xlim(0.01, 10.0)
            plt.yscale("log")
            plt.xscale("log")
            plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                       [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
            plt.legend()
            plt.savefig(
                f"response spectrum-all_rake-{plot_all_rake} Mw-{self.Mw} Rrup-{self.rrup} Vs30-{self.Vs30} fault-type-{self.fault_type_dict[self.rake]} station-{self.station_id}.png",
                dpi=300)
            plt.show()

        # * 2. Mw independent
        Mw_list = [4, 5, 6, 7]
        if avg_station_id:
            for i, _Mw in enumerate(Mw_list):
                total_sta_predict = []
                for sta in tqdm(range(self.station_num)):  # 預測所有station取平均
                    single_sta_predict = []
                    RSCon = xgb.DMatrix(np.array([[np.log(self.Vs30), _Mw,
                                                   np.log(self.rrup), self.rake, sta]]))
                    for _model in booster:
                        predict_value = np.exp(_model.predict(RSCon)) / 980
                        single_sta_predict.append(predict_value)

                    total_sta_predict.append(single_sta_predict)
                plt.plot(self.period_list,
                         np.array(total_sta_predict).mean(axis=0),
                         linewidth='1.2', zorder=20, label=f'Mw:{_Mw}')

        else:
            for i, _Mw in enumerate(Mw_list):
                single_sta_predict = []
                RSCon = xgb.DMatrix(np.array([[np.log(self.Vs30), _Mw,
                                               np.log(self.rrup), self.rake, self.station_id]]))
                for _model in booster:
                    predict_value = np.exp(_model.predict(RSCon)) / 980
                    single_sta_predict.append(predict_value)
                plt.plot(self.period_list,
                         single_sta_predict, label=f'Mw:{_Mw}')

        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.title(
            f"Fault_type = {self.fault_type_dict[self.rake]}, Rrup = {self.rrup}km, Vs30 = {self.Vs30}m/s")
        plt.xlabel("Period(s)", fontsize=12)
        plt.ylabel("PSA(g)", fontsize=12)
        plt.ylim(10e-6, 1)
        plt.xlim(0.01, 10.0)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                   [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
        plt.legend(loc="lower left")
        plt.savefig(
            f"response spectrum-all_rake-{plot_all_rake} avg_station_id-{avg_station_id} Fault_type-{self.fault_type_dict[self.rake]} Rrup-{self.rrup} Vs30-{self.Vs30} station-{self.station_id}.png",
            dpi=300)
        plt.show()


if __name__ == '__main__':

    target = "PGA"
    Mw = 7.65
    Rrup = 50
    Vs30 = 360
    rake = 90
    station_id = 50
    station_id_num = 732  # station 總量

    # ? data preprocess
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_period/TSMIP_FF_{target}.csv")
    TSMIP_all_df = pd.read_csv(f"../../../TSMIP_FF.csv")

    filter = TSMIP_all_df[TSMIP_all_df['eq.type']
                          == "shallow crustal"].reset_index()
    station_order = filter[filter["EQ_ID"] == "1999_0920_1747_16"][[
        "STA_Lon_X", "STA_Lat_Y", "STA_rank", "STA_ID"]]
    index_start = station_order.index[0]
    index_end = station_order.index[-1]+1
    # 1999_0920_1747_16 -> 3061~3460
    # 2009_0817_0005_46 -> 16931~17245
    # 2010_0304_0816_16 -> 18824~19064

    model = dataprocess()
    after_process_ori_data = model.preProcess(TSMIP_df, target, True)

    model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
    original_data = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                       False, *model_feture)

    # ? model predicted
    booster = xgb.Booster()
    booster.load_model(f'../XGB/model/XGB_{target}.json')

    # ? plot figure
    plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
    plot_something.explainable(station_order, original_data[0], model_feture,
                               booster, index_start, index_end)
