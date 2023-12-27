import xgboost as xgb
from modules.process_train import dataprocess
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import shap
sys.path.append("..")
sys.path.append("../modules/gsim")
from tqdm import tqdm
from modules.gsim.chao_2020 import ChaoEtAl2020Asc
from modules.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
from modules.gsim.abrahamson_2014 import AbrahamsonEtAl2014
from modules.gsim.lin_2009 import Lin2009
from modules.gsim.chang_2023 import Chang2023
from modules.gsim.phung_2020 import PhungEtAl2020Asc
from modules.gsim.utils.imt import PGA, SA, PGV
from numpy.lib import recfunctions

class plot_fig:
    """

    Class method for plotting figure.

    """

    def __init__(self, model_name, abbreviation_name, SMOGN_TSMIP, target):
        self.model_name = model_name
        self.abbreviation_name = abbreviation_name
        self.SMOGN_TSMIP = SMOGN_TSMIP
        self.target = target
        self.fault_type_dict = {90: "REV", -90: "NM", 0: "SS"}

    def data_distribution(self, x_total, y_total):
        """

        Plot the train & test data distribution.

        Args:
            x_total ([dataframe]): [original feature data]
            y_total ([dataframe]): [original answer data]
        """

        # Vs30 relationship
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(x_total):  # 計算每個網格中總點數
            x_net = (round(np.exp(x_total[:, 0])[i], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)  # 看是落在哪個x網格
            y_net = (round(y_total[i], 2) - (-1)) / (
                (8 - (-1)) / net)  # 看是落在哪個y網格
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格+=1
            i += 1

        j = 0
        while j < len(x_total):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(np.exp(x_total[:, 0])[j], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)
            y_net = (round(y_total[j], 2) - (-1)) / ((8 - (-1)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料網格的數值(zz值) 用以畫圖
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(np.exp(x_total[:, 0]),
                    y_total,
                    marker='o',
                    facecolors='none',
                    c=color_column,
                    cmap=newcmp,
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.xlabel('Vs30(m/s)', fontsize=12)
        plt.ylabel(f'Measured ln({self.target})(cm/s^2)', fontsize=12)
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

        # Mw relationship
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(x_total):
            x_net = (round(x_total[:, 1][i], 2) - 3) / ((8 - 3) / net)
            y_net = (round(y_total[i], 2) - (-1)) / ((8 - (-1)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(x_total):
            x_net = (round(x_total[:, 1][j], 2) - 3) / ((8 - 3) / net)
            y_net = (round(y_total[j], 2) - (-1)) / ((8 - (-1)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)
        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(x_total[:, 1],
                    y_total,
                    marker='o',
                    facecolors='none',
                    c=color_column,
                    cmap=newcmp,
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.xlabel('Mw', fontsize=12)
        plt.ylabel(f'Measured ln({self.target})(cm/s^2)', fontsize=12)
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

        # Rrup relationship
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(x_total):
            x_net = (round(np.exp(x_total[:, 2])[i], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(y_total[i], 2) - (-1)) / ((8 - (-1)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(x_total):
            x_net = (round(np.exp(x_total[:, 2])[j], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(y_total[j], 2) - (-1)) / ((8 - (-1)) / net)
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
        plt.scatter(np.exp(x_total[:, 2]),
                    y_total,
                    marker='o',
                    facecolors='none',
                    c=color_column,
                    cmap=newcmp,
                    s=8,
                    zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.xlabel('ln(Rrup)(km)', fontsize=12)
        plt.ylabel(f'Measured ln({self.target})(cm/s^2)', fontsize=12)
        plt.xscale("log")
        plt.xlim(1 * 1e0, 1e3)
        plt.xticks([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500
        ], [
            1, '', '', '', '', '', '', '', '', 10, '', '', '', 50, '', '', '',
            '', 100, 200, 300, '', 500
        ])
        plt.title(f'Data Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

    def residual(self, x_total: "ori_feature", y_total: "ori_ans",
                 predict_value: "ori_predicted",
                 ori_full_data: "ori_notsplit_data", score):
        """

        Total residual and standard deviation and inter & intra event residual.
        Calculate residual by original anser and original predicted value.

        """

        # * 1. Vs30 Total Residual
        residual = predict_value - y_total
        residual_121 = []
        residual_199 = []
        residual_398 = []
        residual_794 = []
        residual_1000 = []

        for index, i in enumerate(np.exp(x_total[:, 0])):
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

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(np.exp(x_total[:, 0])[i], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)  # 看是落在哪個x網格
            y_net = (round(residual[i], 2) - (-3)) / (
                (3 - (-3)) / net)  # 看是落在哪個y網格
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格+=1
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(np.exp(x_total[:, 0])[j], 2) - 1e2) / (
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
        plt.scatter(np.exp(x_total[:, 0]),
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
            % (score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict Residual.png',
            dpi=300)
        plt.show()

        # * 2. Mw Toral Residual
        residual = predict_value - y_total
        residual_3_5 = []
        residual_4_5 = []
        residual_5_5 = []
        residual_6_5 = []

        for index, i in enumerate(x_total[:, 1]):
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
            x_net = (round(x_total[:, 1][i], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(x_total[:, 1][j], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(x_total[:, 1], residual,
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
            % (score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict Residual.png',
            dpi=300)
        plt.show()

        # *  3. Rrup Total Residual
        residual = predict_value - y_total
        residual_10 = []
        residual_31 = []
        residual_100 = []
        residual_316 = []

        for index, i in enumerate(np.exp(x_total[:, 2])):
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
            x_net = (round(np.exp(x_total[:, 2])[i], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1
            i += 1

        j = 0
        while j < len(residual):
            x_net = (round(np.exp(x_total[:, 2])[j], 2) - 5 * 1e0) / (
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
        plt.scatter(np.exp(x_total[:, 2]),
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
            % (score, total_std))
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Predict Residual.png',
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
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Total-Residual Distribution.png',
            dpi=300)
        plt.show()
        """

        inter & intra event residual
        
        """

        # 計算inter-event by mean value(共273顆地震)
        originaldata_predicted_result_df = pd.DataFrame(predict_value,
                                                        columns=['predicted'])
        total_data_df = pd.concat(
            [ori_full_data, originaldata_predicted_result_df], axis=1)
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
        inter_mean = round(inter_event['inter_event_residual'].mean(), 2)
        inter_std = round(inter_event['inter_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Inter-event Residual Mean:{inter_mean} Std:{inter_std}'
        )
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Inter-event Residual.png',
            dpi=300)
        plt.show()

        # 計算intra-event by total residual - inter-event residual
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
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        intra_std = round(total_data_df['intra_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean} Std:{intra_std}'
        )
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Intra-event Residual.png',
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
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        intra_std = round(total_data_df['intra_event_residual'].std(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean} Std:{intra_std}'
        )
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Intra-event Residual.png',
            dpi=300)
        plt.show()

    def measured_predict(self, y_test: "ori_ans",
                         predict_value: "ori_predicted",
                         score,
                         lowerbound,
                         higherbound):
        """

        Plot the predict value and true value distribution .

        Args:
            y_test (ori_ans): [original dataset answer]
            predict_value (ori_predicted): [prediction value]
            score ([float]): [R2 score from model]
            lowerbound ([int]): [net lowerbound]
            higherbound ([int]): [net higherbound]
        """
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)
        color_column = []

        i = 0
        while i < len(y_test):
            x_net = (round(y_test[i], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            # +2:因為網格從-2開始打 10:頭減尾8-(-2) 10/net:網格間格距離 x_net:x方向第幾個網格
            y_net = (round(predict_value[i], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            if x_net < net and y_net < net:
                zz[math.floor(x_net), math.floor(y_net)] += 1
            else:
                zz[net-1, net-1] += 1
            i += 1

        j = 0
        while j < len(y_test):
            x_net = (round(y_test[j], 2) - lowerbound) / \
                ((higherbound - lowerbound) / net)
            y_net = (round(predict_value[j], 2) - lowerbound) / \
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
        plt.scatter(y_test, predict_value, s=8,
                    c=color_column, cmap=newcmp, zorder=10)
        x_line = [lowerbound, higherbound]
        y_line = [lowerbound, higherbound]
        plt.plot(x_line, y_line, 'r--', alpha=0.5, zorder=10)
        plt.xlabel(f'Measured ln({self.target})(cm/s^2)', fontsize=12)
        plt.ylabel(f'Predict ln({self.target})(cm/s^2)', fontsize=12)
        plt.ylim(lowerbound, higherbound)
        plt.xlim(lowerbound, higherbound)
        plt.title(
            f'{self.SMOGN_TSMIP} {self.abbreviation_name} Measured Predicted Distribution'
        )
        plt.text(higherbound-4, lowerbound+2, f"R2 score = {round(score,2)}")
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Measured Predicted Comparison.png',
            dpi=300)
        plt.show()

    def distance_scaling(
            self,
            DSC_df,
            station_id_num: "station總量",
            plot_all_sta,
            x_total: "ori_feature",
            y_total: "ori_ans",
            model_path):
        """

        Compute distance scaling figure follow condition given by ourself.

        Args:
            DSC_df ([dataframe]): [the condition given by ourself]
            station_id_num (station): [total station number]
            plot_all_sta ([bool]): [to decide if we want to plot all station]
            x_total (ori_feature): [feature]
            y_total (ori_ans): [value]
            model_path ([str]): [the place which the model be stored]
        """
        dataLen = 17  # rrup 總點位
        total = np.array([0]*dataLen)

        # * calculate Chang2023 total station value
        ch_mean = [[0] * dataLen] * station_id_num
        ch_sig = [[0] * dataLen] * station_id_num
        ch_tau = [[0] * dataLen] * station_id_num
        ch_phi = [[0] * dataLen] * station_id_num
        for i in tqdm(range(station_id_num)):
            ctx = DSC_df[DSC_df['sta_id'] == i+1].to_records()
            ctx = recfunctions.drop_fields(
                ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
            ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                              ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                              ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
                              ('ry0', '<f8'), ('width',
                                               '<f8'), ('vs30measured', 'bool'),
                              ('sta_id', '<i8'), ('hypo_depth', '<f8'), ('z2pt5', '<f8')])
            ctx = ctx.view(np.recarray)
            imts = [PGA()]
            chang = Chang2023(model_path)
            ch_mean[i], ch_sig[i], ch_tau[i], ch_phi[i] = chang.compute(
                ctx, imts, [ch_mean[i]], [ch_sig[i]], [ch_tau[i]], [ch_phi[i]])
            if plot_all_sta:
                ch_mean_copy = np.exp(ch_mean[i][0].copy())
                plt.plot(ctx['rrup'], ch_mean_copy, 'lightgrey', linewidth='0.4', zorder=5)
            else:
                total = total + np.exp(ch_mean[i][0])

        if not plot_all_sta:
            total_station_mean = total / station_id_num
            plt.plot(ctx['rrup'], total_station_mean, 'r',
                     linewidth='1.6', label="This study avg", zorder=20)

        # * 2.others GMM
        ctx = DSC_df[DSC_df['sta_id'] == 1].to_records()  # 其餘GMM用不著sta_id
        ctx = recfunctions.drop_fields(
            ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
        ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                          ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                          ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
                          ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
                          ('sta_id', '<i8'), ('hypo_depth', '<f8'), ('z2pt5', '<f8')])
        ctx = ctx.view(np.recarray)
        imts = [PGA()]

        phung = PhungEtAl2020Asc()
        ph_mean = [[0] * dataLen]
        ph_sig = [[0] * dataLen]
        ph_tau = [[0] * dataLen]
        ph_phi = [[0] * dataLen]
        ph_mean, ph_sig, ph_tau, ph_phi = phung.compute(
            ctx, imts, ph_mean, ph_sig, ph_tau, ph_phi)
        ph_mean = np.exp(ph_mean)

        lin = Lin2009()
        lin_mean = [[0] * dataLen]
        lin_sig = [[0] * dataLen]
        lin_tau = [[0] * dataLen]
        lin_phi = [[0] * dataLen]
        lin_mean, lin_sig = lin.compute(
            ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
        lin_mean = np.exp(lin_mean)

        abrahamson = AbrahamsonEtAl2014()
        abr_mean = [[0] * dataLen]
        abr_sig = [[0] * dataLen]
        abr_tau = [[0] * dataLen]
        abr_phi = [[0] * dataLen]
        abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(
            ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
        abr_mean = np.exp(abr_mean)

        campbell = CampbellBozorgnia2014()
        cam_mean = [[0] * dataLen]
        cam_sig = [[0] * dataLen]
        cam_tau = [[0] * dataLen]
        cam_phi = [[0] * dataLen]
        cam_mean, cam_sig, cam_tau, cam_phi = campbell.compute(
            ctx, imts, cam_mean, cam_sig, cam_tau, cam_phi)
        cam_mean = np.exp(cam_mean)

        choa = ChaoEtAl2020Asc()
        choa_mean = [[0] * dataLen]
        choa_sig = [[0] * dataLen]
        choa_tau = [[0] * dataLen]
        choa_phi = [[0] * dataLen]
        choa_mean, choa_sig, choa_tau, choa_phi = choa.compute(
            ctx, imts, choa_mean, choa_sig, choa_tau, choa_phi)
        choa_mean = np.exp([choa_mean])

        # * 3.plt figure
        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.scatter(np.exp(x_total[:, 2]),
                    np.exp(y_total) / 980,
                    marker='o',
                    facecolors='none',
                    s=8,
                    color='grey',
                    label='data')
        # plt.plot(ctx['rrup'], ch_mean[0] + ch_sig[0], 'b--')
        # plt.plot(ctx['rrup'], ch_mean[0] - ch_sig[0], 'b--')
        plt.plot(ctx['rrup'], ph_mean[0], 'orange',
                 linewidth='1', label="Phung2020", zorder=10)
        # plt.plot(ctx['rrup'], ph_mean[0] + ph_sig[0], 'r--')
        # plt.plot(ctx['rrup'], ph_mean[0] - ph_sig[0], 'r--')
        plt.plot(ctx['rrup'], lin_mean[0], 'g',
                 linewidth='1', label="Lin2009", zorder=10)
        # plt.plot(ctx['rrup'], lin_mean[0] + lin_sig[0], 'g--')
        # plt.plot(ctx['rrup'], lin_mean[0] - lin_sig[0], 'g--')
        plt.plot(ctx['rrup'], abr_mean[0], 'b',
                 linewidth='1', label="Abrahamson2014", zorder=10)
        # plt.plot(ctx['rrup'], abr_mean[0] + abr_sig[0], 'r--')
        # plt.plot(ctx['rrup'], abr_mean[0] - abr_sig[0], 'r--')
        plt.plot(ctx['rrup'], cam_mean[0], 'yellow',
                 linewidth='1', label="CampbellBozorgnia2014", zorder=10)
        # plt.plot(ctx['rrup'], cam_mean[0] + choa_sig[0], 'r--')
        # plt.plot(ctx['rrup'], cam_mean[0] - choa_sig[0], 'r--')
        plt.plot(ctx['rrup'], choa_mean[0], 'pink',
                 linewidth='1', label="ChaoEtAl2020Asc", zorder=10)
        # plt.plot(ctx['rrup'], choa_mean[0] + choa_sig[0], 'r--')
        # plt.plot(ctx['rrup'], choa_mean[0] - choa_sig[0], 'r--')
        plt.xlabel(f'Rrup(km)')
        plt.ylabel(f'{self.target}(g)')
        plt.title(
            f"Mw = {ctx['mag'][0]}, Vs30 = {ctx['vs30'][0]}m/s  Fault = {self.fault_type_dict[ctx['rake'][0]]}")
        plt.ylim(10e-5, 10)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
                   [0.1, 0.5, 1, 10, 50, 100, 200, 300])
        plt.legend()
        plt.savefig(
            f"distance scaling Mw-{ctx['mag'][0]} Vs30-{ctx['vs30'][0]} fault-type-{self.fault_type_dict[ctx['rake'][0]]} global-{plot_all_sta}.jpg", dpi=300)
        plt.show()

    def explainable(self, x_total: "ori_total_feature", model_feture, ML_model,
                    seed):
        """

        This function shows explanation wihch include global explaination and local explaination of the model in the given target .

        Args:
            x_total (ori_total_feature): [original feature data]
            model_feture ([list]): [input parameters]
            ML_model ([model]): [the model from sklearn or other package]
            seed ([int]): [random seed number]
        """
        df = pd.DataFrame(x_total, columns=model_feture)
        explainer = shap.Explainer(ML_model)
        shap_values = explainer(df)

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

        #! Local Explainable
        # waterfall
        fig = plt.figure()
        shap.plots.waterfall(shap_values[seed],
                             show=False)  # 單筆資料解釋:第seed筆資料解釋
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_waterfall_{seed}_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # force plot
        shap.initjs()
        shap.force_plot(explainer.expected_value,
                        shap_values.values[seed, :],
                        df.iloc[seed, :],
                        show=False,
                        matplotlib=True)
        plt.savefig(f"force_plot_{seed}_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        # 強制刷新緩存
        fig.canvas.draw()

    def respond_spectrum(self, Vs30, Mw, Rrup, rake, station_rank, local,
                         *args: "model"):
        """

        This function is called to plot respond spectrum .

        Args:
            Vs30 ([float]): [Vs30 value]
            Mw ([int]): [Mw value]
            Rrup ([float]): [Rrup value]
            rake ([float]): [rake value]
            station_rank ([int]): [station number value]
            local ([bool]): [to decide plot local or global figure]
        """
        booster_PGA = xgb.Booster()
        booster_PGA.load_model(args[0])
        booster_PGV = xgb.Booster()
        booster_PGV.load_model(args[1])
        booster_Sa001 = xgb.Booster()
        booster_Sa001.load_model(args[2])
        booster_Sa005 = xgb.Booster()
        booster_Sa005.load_model(args[3])
        booster_Sa01 = xgb.Booster()
        booster_Sa01.load_model(args[4])
        booster_Sa02 = xgb.Booster()
        booster_Sa02.load_model(args[5])
        booster_Sa05 = xgb.Booster()
        booster_Sa05.load_model(args[6])
        booster_Sa10 = xgb.Booster()
        booster_Sa10.load_model(args[7])
        booster_Sa30 = xgb.Booster()
        booster_Sa30.load_model(args[8])
        booster_Sa40 = xgb.Booster()
        booster_Sa40.load_model(args[9])
        booster_Sa100 = xgb.Booster()
        booster_Sa100.load_model(args[10])

        # * 1. focal.type independent
        if local == True:
            RSCon = xgb.DMatrix(
                np.array([[np.log(Vs30), Mw,
                           np.log(Rrup), rake, station_rank]]))
            Sa001_predict = np.exp(booster_Sa001.predict(RSCon)) / 980
            Sa005_predict = np.exp(booster_Sa005.predict(RSCon)) / 980
            Sa01_predict = np.exp(booster_Sa01.predict(RSCon)) / 980
            Sa02_predict = np.exp(booster_Sa02.predict(RSCon)) / 980
            Sa05_predict = np.exp(booster_Sa05.predict(RSCon)) / 980
            Sa10_predict = np.exp(booster_Sa10.predict(RSCon)) / 980
            Sa30_predict = np.exp(booster_Sa30.predict(RSCon)) / 980
            Sa40_predict = np.exp(booster_Sa40.predict(RSCon)) / 980
            Sa100_predict = np.exp(booster_Sa100.predict(RSCon)) / 980
            plt.grid(which="both",
                     axis="both",
                     linestyle="-",
                     linewidth=0.5,
                     alpha=0.5)
            plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
                Sa001_predict[0], Sa005_predict[0], Sa01_predict[0],
                Sa02_predict[0], Sa05_predict[0], Sa10_predict[0],
                Sa30_predict[0], Sa40_predict[0], Sa100_predict[0]
            ], label=self.fault_type_dict[rake])
            plt.title(f"Mw = {Mw}, Rrup = {Rrup}km, Vs30 = {Vs30}m/s")
            plt.xlabel("Period(s)")
            plt.ylabel("PSA(g)")
            plt.ylim(10e-6, 1)
            plt.xlim(0.01, 10.0)
            plt.yscale("log")
            plt.xscale("log")
            plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                       [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
            plt.legend()
            plt.savefig(
                f"response spectrum-local Mw-{Mw} Rrup-{Rrup} Vs30-{Vs30} fault-type-{self.fault_type_dict[rake]} station-{station_rank}.png",
                dpi=300)
            plt.show()

        else:
            for _rake in self.fault_type_dict:
                RSCon = xgb.DMatrix(
                    np.array([[np.log(Vs30), Mw,
                               np.log(Rrup), _rake, station_rank]]))
                Sa001_predict = np.exp(booster_Sa001.predict(RSCon)) / 980
                Sa005_predict = np.exp(booster_Sa005.predict(RSCon)) / 980
                Sa01_predict = np.exp(booster_Sa01.predict(RSCon)) / 980
                Sa02_predict = np.exp(booster_Sa02.predict(RSCon)) / 980
                Sa05_predict = np.exp(booster_Sa05.predict(RSCon)) / 980
                Sa10_predict = np.exp(booster_Sa10.predict(RSCon)) / 980
                Sa30_predict = np.exp(booster_Sa30.predict(RSCon)) / 980
                Sa40_predict = np.exp(booster_Sa40.predict(RSCon)) / 980
                Sa100_predict = np.exp(booster_Sa100.predict(RSCon)) / 980
                plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
                    Sa001_predict[0], Sa005_predict[0], Sa01_predict[0],
                    Sa02_predict[0], Sa05_predict[0], Sa10_predict[0],
                    Sa30_predict[0], Sa40_predict[0], Sa100_predict[0]
                ], label=self.fault_type_dict[_rake])

            plt.grid(which="both",
                     axis="both",
                     linestyle="-",
                     linewidth=0.5,
                     alpha=0.5)
            plt.title(f"Mw = {Mw}, Rrup = {Rrup}km, Vs30 = {Vs30}m/s")
            plt.xlabel("Period(s)")
            plt.ylabel("PSA(g)")
            plt.ylim(10e-6, 1)
            plt.xlim(0.01, 10.0)
            plt.yscale("log")
            plt.xscale("log")
            plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                       [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
            plt.legend()
            plt.savefig(
                f"response spectrum-global Mw-{Mw} Rrup-{Rrup} Vs30-{Vs30} station-{station_rank}.png",
                dpi=300)
            plt.show()

        # * 2. Mw independent
        Mw_list = [4, 5, 6, 7]
        for _Mw in Mw_list:
            RSCon = xgb.DMatrix(
                np.array([[np.log(Vs30), _Mw,
                           np.log(Rrup), rake, station_rank]]))
            Sa001_predict = np.exp(booster_Sa001.predict(RSCon)) / 980
            Sa005_predict = np.exp(booster_Sa005.predict(RSCon)) / 980
            Sa01_predict = np.exp(booster_Sa01.predict(RSCon)) / 980
            Sa02_predict = np.exp(booster_Sa02.predict(RSCon)) / 980
            Sa05_predict = np.exp(booster_Sa05.predict(RSCon)) / 980
            Sa10_predict = np.exp(booster_Sa10.predict(RSCon)) / 980
            Sa30_predict = np.exp(booster_Sa30.predict(RSCon)) / 980
            Sa40_predict = np.exp(booster_Sa40.predict(RSCon)) / 980
            Sa100_predict = np.exp(booster_Sa100.predict(RSCon)) / 980
            plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
                Sa001_predict[0], Sa005_predict[0], Sa01_predict[0],
                Sa02_predict[0], Sa05_predict[0], Sa10_predict[0],
                Sa30_predict[0], Sa40_predict[0], Sa100_predict[0]
            ], label=f'Mw:{_Mw}')

        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.title(
            f"Fault_type = {self.fault_type_dict[rake]}, Rrup = {Rrup}km, Vs30 = {Vs30}m/s")
        plt.xlabel("Period(s)")
        plt.ylabel("PSA(g)")
        plt.ylim(10e-6, 1)
        plt.xlim(0.01, 10.0)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                   [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0])
        plt.legend()
        plt.savefig(
            f"response spectrum-global Fault_type-{self.fault_type_dict[rake]} Rrup-{Rrup} Vs30-{Vs30} station-{station_rank}.png",
            dpi=300)
        plt.show()


if __name__ == '__main__':
    TSMIP_smogn_df = pd.read_csv("../../../TSMIP_smogn_sta.csv")
    TSMIP_df = pd.read_csv("../../../TSMIP_FF_copy.csv")
    model = dataprocess()
    after_process_data = model.preprocess(TSMIP_smogn_df)
    result_list = model.split_dataset(TSMIP_smogn_df, 'lnPGA(gal)', True,
                                      'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                      'STA_Lon_X', 'STA_Lat_Y')
    score, feature_importances, fit_time, final_predict = model.training(
        "XGB", result_list[0], result_list[1], result_list[2], result_list[3])

    plot_something = plot_fig("XGBooster", "XGB", "TSMIP")
    plot_something.train_test_distribution(result_list[1], result_list[3],
                                           final_predict, fit_time, score)
    plot_something.residual(result_list[1], result_list[3], final_predict,
                            score)
    plot_something.measured_predict(result_list[3], final_predict, score)
    c = result_list[1].transpose(1, 0)
    plot_something.distance_scaling(c[2], final_predict, score)

    # 線性的distance_scaling
    # c = result_list[1].transpose(1, 0)
    # concat_data = np.concatenate((np.array([c[2]]), np.array([final_predict])),
    #                              axis=0).transpose(1, 0)
    # concat_data_df = pd.DataFrame(concat_data, columns=['dist', 'predict'])
    # concat_data_df = concat_data_df.sort_values(by=['dist'])
    # concat_data = np.array(concat_data_df).transpose(1, 0)
    # plot_something.distance_scaling(concat_data[0], concat_data[1], score)
