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
        unique_color = set(x_total[:, 3])
        color_map = {-90.0: ["r","NM"], 0.0: ["g","SS"], 90.0: ["b","REV"]}

        # # Vs30 Mw relationship

        plt.grid(linestyle=':', color='darkgrey', zorder=0)
        plt.scatter(np.exp(x_total[:, 0]),
                    x_total[:, 1],
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
            indices = x_total[:, 3] == color
            plt.scatter(x_total[indices, -1], x_total[indices, 1],
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
            indices = x_total[:, 3] == color
            plt.scatter(np.exp(x_total[indices, 2]), x_total[indices, 1],
                        facecolors='none', c=color_map[color][0],
                        s=8, zorder=10, label=color_map[color][1])
        plt.legend(loc="lower left")
        plt.xlabel('Rupture Distance, Rrup(km)', fontsize=12)
        plt.ylabel(f'Moment Magnitude, Mw', fontsize=12)
        plt.xscale("log")
        plt.xlim(1 * 1e0, 7* 1e2)
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

        counter = Counter(x_total[:, 3])
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

    def residual(self, x_total, y_total, predict_value, ori_full_data, score):
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

    def measured_predict(self, y_test,
                         predict_value,
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
            Vs30,
            rake,
            station_id_num: "station總量",
            plot_all_sta,
            station_id,
            total_data,
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

        #* comparsion ohter GMMs

        # dataLen = 17  # rrup 總點位
        # total = np.array([0]*dataLen)
        # ch_mean = [[0] * dataLen] * station_id_num
        # ch_sig = [[0] * dataLen] * station_id_num
        # ch_tau = [[0] * dataLen] * station_id_num
        # ch_phi = [[0] * dataLen] * station_id_num

        # #calculate Chang2023 total station value
        # for i in tqdm(range(station_id_num)):
        #     ctx = DSC_df[DSC_df['sta_id'] == i+1].to_records()
        #     ctx = recfunctions.drop_fields(
        #         ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
        #     ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
        #                       ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
        #                       ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
        #                       ('ry0', '<f8'), ('width',
        #                                        '<f8'), ('vs30measured', 'bool'),
        #                       ('sta_id', '<i8'), ('hypo_depth', '<f8'), ('z2pt5', '<f8')])
        #     ctx = ctx.view(np.recarray)
        #     imts = [PGA()]
        #     chang = Chang2023(model_path)
        #     ch_mean[i], ch_sig[i], ch_tau[i], ch_phi[i] = chang.compute(
        #         ctx, imts, [ch_mean[i]], [ch_sig[i]], [ch_tau[i]], [ch_phi[i]])
        #     if plot_all_sta:
        #         ch_mean_copy = np.exp(ch_mean[i][0].copy())
        #         plt.plot(ctx['rrup'], ch_mean_copy,
        #                  'r', linewidth='0.4', zorder=5)
        #     else:
        #         total = total + np.exp(ch_mean[i][0])
        # if not plot_all_sta:
        #     total_station_mean = total / station_id_num
        #     plt.plot(ctx['rrup'], total_station_mean, 'r',
        #              linewidth='1.6', label="This study avg", zorder=20)

        # #others GMM
        # ctx = DSC_df[DSC_df['sta_id'] == 1].to_records()  # 其餘GMM用不著sta_id
        # ctx = recfunctions.drop_fields(
        #     ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
        # ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
        #                   ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
        #                   ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
        #                   ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
        #                   ('sta_id', '<i8'), ('hypo_depth', '<f8'), ('z2pt5', '<f8')])
        # ctx = ctx.view(np.recarray)
        # imts = [PGA()]
        # phung = PhungEtAl2020Asc()
        # ph_mean = [[0] * dataLen]
        # ph_sig = [[0] * dataLen]
        # ph_tau = [[0] * dataLen]
        # ph_phi = [[0] * dataLen]
        # ph_mean, ph_sig, ph_tau, ph_phi = phung.compute(
        #     ctx, imts, ph_mean, ph_sig, ph_tau, ph_phi)
        # ph_mean = np.exp(ph_mean)
        # lin = Lin2009()
        # lin_mean = [[0] * dataLen]
        # lin_sig = [[0] * dataLen]
        # lin_tau = [[0] * dataLen]
        # lin_phi = [[0] * dataLen]
        # lin_mean, lin_sig = lin.compute(
        #     ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
        # lin_mean = np.exp(lin_mean)
        # abrahamson = AbrahamsonEtAl2014()
        # abr_mean = [[0] * dataLen]
        # abr_sig = [[0] * dataLen]
        # abr_tau = [[0] * dataLen]
        # abr_phi = [[0] * dataLen]
        # abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(
        #     ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
        # abr_mean = np.exp(abr_mean)
        # campbell = CampbellBozorgnia2014()
        # cam_mean = [[0] * dataLen]
        # cam_sig = [[0] * dataLen]
        # cam_tau = [[0] * dataLen]
        # cam_phi = [[0] * dataLen]
        # cam_mean, cam_sig, cam_tau, cam_phi = campbell.compute(
        #     ctx, imts, cam_mean, cam_sig, cam_tau, cam_phi)
        # cam_mean = np.exp(cam_mean)
        # choa = ChaoEtAl2020Asc()
        # choa_mean = [[0] * dataLen]
        # choa_sig = [[0] * dataLen]
        # choa_tau = [[0] * dataLen]
        # choa_phi = [[0] * dataLen]
        # choa_mean, choa_sig, choa_tau, choa_phi = choa.compute(
        #     ctx, imts, choa_mean, choa_sig, choa_tau, choa_phi)
        # choa_mean = np.exp([choa_mean])

        # plt.grid(which="both",
        #          axis="both",
        #          linestyle="-",
        #          linewidth=0.5,
        #          alpha=0.5)
        # plt.scatter(np.exp(total_data[0][0][:, 2]),
        #             np.exp(total_data[0][1]) / 980,
        #             marker='o',
        #             facecolors='none',
        #             s=2,
        #             color='grey',
        #             label='data',
        #             zorder=20)
        # # plt.plot(ctx['rrup'], ch_mean[0] + ch_sig[0], 'b--')
        # # plt.plot(ctx['rrup'], ch_mean[0] - ch_sig[0], 'b--')
        # plt.plot(ctx['rrup'], ph_mean[0], 'orange',
        #          linewidth='1', label="Phung2020", zorder=10)
        # # plt.plot(ctx['rrup'], ph_mean[0] + ph_sig[0], 'r--')
        # # plt.plot(ctx['rrup'], ph_mean[0] - ph_sig[0], 'r--')
        # plt.plot(ctx['rrup'], lin_mean[0], 'g',
        #          linewidth='1', label="Lin2009", zorder=10)
        # # plt.plot(ctx['rrup'], lin_mean[0] + lin_sig[0], 'g--')
        # # plt.plot(ctx['rrup'], lin_mean[0] - lin_sig[0], 'g--')
        # plt.plot(ctx['rrup'], abr_mean[0], 'b',
        #          linewidth='1', label="Abrahamson2014", zorder=10)
        # # plt.plot(ctx['rrup'], abr_mean[0] + abr_sig[0], 'r--')
        # # plt.plot(ctx['rrup'], abr_mean[0] - abr_sig[0], 'r--')
        # plt.plot(ctx['rrup'], cam_mean[0], 'yellow',
        #          linewidth='1', label="CampbellBozorgnia2014", zorder=10)
        # # plt.plot(ctx['rrup'], cam_mean[0] + choa_sig[0], 'r--')
        # # plt.plot(ctx['rrup'], cam_mean[0] - choa_sig[0], 'r--')
        # plt.plot(ctx['rrup'], choa_mean[0], 'pink',
        #          linewidth='1', label="ChaoEtAl2020Asc", zorder=10)
        # # plt.plot(ctx['rrup'], choa_mean[0] + choa_sig[0], 'r--')
        # # plt.plot(ctx['rrup'], choa_mean[0] - choa_sig[0], 'r--')
        # plt.xlabel(f'Rrup(km)')
        # plt.ylabel(f'{self.target}(g)')
        # plt.title(
        #     f"Mw = {ctx['mag'][0]}, Vs30 = {ctx['vs30'][0]}m/s  Fault = {self.fault_type_dict[ctx['rake'][0]]}")
        # plt.ylim(10e-4, 5)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
        #            [0.1, 0.5, 1, 10, 50, 100, 200, 300])
        # plt.legend()
        # plt.savefig(
        #     f"distance scaling Mw-{ctx['mag'][0]} Vs30-{ctx['vs30'][0]} fault-type-{self.fault_type_dict[ctx['rake'][0]]} global-{plot_all_sta}.jpg", dpi=300)
        # plt.show()

        #* comparsion different Mw
        booster_PGA = xgb.Booster()
        booster_PGA.load_model(model_path)

        Mw_list = [4,5,6,7]
        color = ["lightblue","lightsalmon","lightgreen","lightcoral"]

        if plot_all_sta:
            for i, Mw in enumerate(Mw_list):
                single_sta_predict = []
                for rrup in [0.1, 0.5, 1, 10, 50, 100, 200, 300]:
                    RSCon = xgb.DMatrix(
                        np.array([[np.log(Vs30), Mw,
                            np.log(rrup), rake, station_id]]))
                    single_sta_predict.append(np.exp(booster_PGA.predict(RSCon)) / 980)
                plt.scatter(np.exp(total_data[i+1][0][:, 2]),
                    np.exp(total_data[i+1][1]) / 980,
                    marker='o',
                    facecolors='none',
                    s=2,
                    c=color[i],
                    zorder=5)
                plt.plot([0.1, 0.5, 1, 10, 50, 100, 200, 300], single_sta_predict,
                        linewidth='1.2', zorder=20, label=f"Mw:{Mw_list[i]}")
        else:
            for i, Mw in enumerate(Mw_list):
                total_sta_predict = []
                for sta in tqdm(range(station_id_num)): # 預測所有station取平均
                    single_sta_predict = []
                    for rrup in [0.1, 0.5, 1, 10, 50, 100, 200, 300]:
                        RSCon = xgb.DMatrix(
                            np.array([[np.log(Vs30), Mw,
                                np.log(rrup), 90, sta]]))
                        single_sta_predict.append(np.exp(booster_PGA.predict(RSCon)) / 980)
                    total_sta_predict.append(single_sta_predict)
                plt.scatter(np.exp(total_data[i+1][0][:, 2]),
                    np.exp(total_data[i+1][1]) / 980,
                    marker='o',
                    facecolors='none',
                    s=2,
                    c=color[i],
                    zorder=5)
                plt.plot([0.1, 0.5, 1, 10, 50, 100, 200, 300], np.array(total_sta_predict).mean(axis=0),
                        linewidth='1.2', zorder=20, label=f"Mw:{Mw_list[i]}")

        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.xlabel(f'Rrup(km)')
        plt.ylabel(f'{self.target}(g)')
        plt.title(
            f"Vs30 = {Vs30}m/s  Fault = {self.fault_type_dict[rake]}")
        plt.ylim(10e-4, 5)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
                   [0.1, 0.5, 1, 10, 50, 100, 200, 300])
        plt.legend(loc="lower left")
        plt.savefig(
            f"distance scaling Vs30-{Vs30} fault-type-{self.fault_type_dict[rake]} global-{plot_all_sta} station_id-{station_id}.jpg", dpi=300)
        plt.show()

    def explainable(self, all_df, x_total, model_feture, ML_model, seed, index_start, index_end):
        """

        This function shows explanation wihch include global explaination and local explaination of the model in the given target .

        Args:
            all_df (all dataset): [all dataset]
            x_total (ori_total_feature): [original feature data]
            model_feture ([list]): [input parameters]
            ML_model ([model]): [the model from sklearn or other package]
            seed ([int]): [random seed number]
        """
        df = pd.DataFrame(x_total, columns=model_feture)
        explainer = shap.Explainer(ML_model)
        shap_values = explainer(df)

        # #! Global Explainable
        # summary
        fig = plt.figure()
        shap.summary_plot(shap_values, df, show=True)
        # plt.savefig(f"summary_plot_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # # bar plot
        # fig = plt.figure()
        # shap.plots.bar(shap_values, show=False)
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.savefig(f"shap_bar_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # # scatter plot
        # fig = plt.figure()
        # shap.plots.scatter(shap_values[:, "MW"],
        #                    color=shap_values[:, "lnRrup"],
        #                    show=False)
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.savefig(f"shap_scatter_Mw_lnRrup_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # fig = plt.figure()
        # shap.plots.scatter(shap_values[:, "lnRrup"],
        #                    color=shap_values[:, "MW"],
        #                    show=False)
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.savefig(f"shap_scatter_lnRrup_Mw_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # #! Local Explainable
        # # waterfall
        # fig = plt.figure()
        # shap.plots.waterfall(shap_values[seed],
        #                      show=False)  # 單筆資料解釋:第seed筆資料解釋
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'
        # plt.savefig(f"shap_waterfall_{seed}_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # # force plot
        # shap.initjs()
        # shap.force_plot(explainer.expected_value,
        #                 shap_values.values[seed, :],
        #                 df.iloc[seed, :],
        #                 show=False,
        #                 matplotlib=True)
        # plt.savefig(f"force_plot_{seed}_{self.target}.jpg",
        #             bbox_inches='tight',
        #             dpi=300)

        # # 強制刷新緩存
        # fig.canvas.draw()

        plt.scatter(all_df["STA_Lon_X"],
                all_df["STA_Lat_Y"],
                c=np.sum(shap_values.values[index_start:index_end],axis=1) \
                    + shap_values.base_values[index_start],
                cmap='cool',
                s=8,
                zorder=10)
        cbar = plt.colorbar(extend='both', label='number value')
        cbar.set_label('number value', fontsize=12)
        plt.xlim(119.4,122.3)
        plt.ylim(21.8,25.5)
        plt.xlabel('longitude', fontsize=12)
        plt.ylabel('latitude', fontsize=12)
        plt.title('TSMIP Station id located')
        plt.show()


        return shap_values

    def respond_spectrum(self, Vs30, Mw, Rrup, rake, station_id, station_id_num,
                        plot_all_rake, plot_all_sta, *args: "model"):
        """

        This function is called to plot respond spectrum .

        Args:
            total_data ([list]): [total filter by Mw dataset]
            Vs30 ([float]): [Vs30 value]
            Mw ([int]): [Mw value]
            Rrup ([float]): [Rrup value]
            rake ([float]): [rake value]
            station_id ([int]): [station number value]
            station_id_num ([int]): [all station number]
            plot_all_rake ([bool]): [to decide if plot all rake figure]
            plot_all_sta ([bool]): [to decide if plot all station figure]
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
        if plot_all_rake == True:
            RSCon = xgb.DMatrix(
                np.array([[np.log(Vs30), Mw,
                           np.log(Rrup), rake, station_id]]))
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
                f"response spectrum-local Mw-{Mw} Rrup-{Rrup} Vs30-{Vs30} fault-type-{self.fault_type_dict[rake]} station-{station_id}.png",
                dpi=300)
            plt.show()

        else:
            for _rake in self.fault_type_dict:
                RSCon = xgb.DMatrix(
                    np.array([[np.log(Vs30), Mw,
                               np.log(Rrup), _rake, station_id]]))
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
                f"response spectrum-global Mw-{Mw} Rrup-{Rrup} Vs30-{Vs30} station-{station_id}.png",
                dpi=300)
            plt.show()

        # * 2. Mw independent
        Mw_list = [4,5,6,7]
        if plot_all_sta:
            for i, _Mw in enumerate(Mw_list):
                total_sta_predict = []
                for sta in tqdm(range(station_id_num)): # 預測所有station取平均
                    single_sta_predict = []
                    RSCon = xgb.DMatrix(
                        np.array([[np.log(Vs30), _Mw,
                                np.log(Rrup), rake, sta]]))
                    Sa001_predict = np.exp(booster_Sa001.predict(RSCon)) / 980
                    Sa005_predict = np.exp(booster_Sa005.predict(RSCon)) / 980
                    Sa01_predict = np.exp(booster_Sa01.predict(RSCon)) / 980
                    Sa02_predict = np.exp(booster_Sa02.predict(RSCon)) / 980
                    Sa05_predict = np.exp(booster_Sa05.predict(RSCon)) / 980
                    Sa10_predict = np.exp(booster_Sa10.predict(RSCon)) / 980
                    Sa30_predict = np.exp(booster_Sa30.predict(RSCon)) / 980
                    Sa40_predict = np.exp(booster_Sa40.predict(RSCon)) / 980
                    Sa100_predict = np.exp(booster_Sa100.predict(RSCon)) / 980
                    single_sta_predict.append(Sa001_predict)
                    single_sta_predict.append(Sa005_predict)
                    single_sta_predict.append(Sa01_predict)
                    single_sta_predict.append(Sa02_predict)
                    single_sta_predict.append(Sa05_predict)
                    single_sta_predict.append(Sa10_predict)
                    single_sta_predict.append(Sa30_predict)
                    single_sta_predict.append(Sa40_predict)
                    single_sta_predict.append(Sa100_predict)

                total_sta_predict.append(single_sta_predict)
                plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0],
                         np.array(total_sta_predict).mean(axis=0),
                         linewidth='1.2', zorder=20, label=f'Mw:{_Mw}')
            
        else:
            for i, _Mw in enumerate(Mw_list):
                RSCon = xgb.DMatrix(
                    np.array([[np.log(Vs30), _Mw,
                            np.log(Rrup), rake, station_id]]))
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
            f"response spectrum-global plot_all_sta-{plot_all_sta} Fault_type-{self.fault_type_dict[rake]} Rrup-{Rrup} Vs30-{Vs30} station-{station_id}.png",
            dpi=300)
        plt.show()


if __name__ == '__main__':
    
    target = "PGA"
    Mw = 7.65
    Rrup = 50
    Vs30 = 360
    rake = 90
    station_id = 50
    station_id_num = 732 # station 總量
    seed = 12974

    #? data preprocess
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_period/TSMIP_FF_{target}.csv")
    TSMIP_all_df = pd.read_csv(f"../../../TSMIP_FF.csv")

    filter = TSMIP_all_df[TSMIP_all_df['eq.type'] == "shallow crustal"].reset_index()
    station_order = filter[filter["EQ_ID"] == "1999_0920_1747_16"][["STA_Lon_X","STA_Lat_Y","STA_rank"]]
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
    
    #? model predicted
    booster = xgb.Booster()
    booster.load_model(f'../XGB/model/XGB_{target}.json')

    #? plot figure
    plot_something = plot_fig("XGBooster", "XGB", "SMOGN", target)
    shap_values=plot_something.explainable(station_order, original_data[0], model_feture,
                                            booster, seed, index_start, index_end)