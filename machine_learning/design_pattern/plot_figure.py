import pandas as pd
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
import os
import shap
import pickle
# append the path of the
# parent directory
sys.path.append("..")
from design_pattern.process_train import dataprocess


class plot_fig:

    def __init__(self, model_name, abbreviation_name, SMOGN_TSMIP, target):
        self.model_name = model_name
        self.abbreviation_name = abbreviation_name
        self.SMOGN_TSMIP = SMOGN_TSMIP
        self.target = target

    def predicted_distribution(self, x_test, y_test, predict_value, fit_time,
                               score):
        """ 
        
        train & test distribution
        
        """

        # 畫 Vs30 and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 0],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='original value')  #數據點
        plt.scatter(x_test[:,0], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'predicted value (accuracy: %.2f)' % (score)) #迴歸線
        plt.xlabel('Vs30(m/s)')
        plt.ylabel(f'Predicted ln({self.target})(cm/s^2)')
        plt.title(f'{self.model_name} Predicted Distribution')
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

        # 畫 Mw and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 1],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='original value')  #數據點
        plt.scatter(x_test[:,1], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'predicted value (accuracy: %.2f)' % (score)) #迴歸線
        plt.xlabel('Mw')
        plt.ylabel(f'Predicted ln({self.target})(cm/s^2)')
        plt.title(f'{self.model_name} Predicted Distribution')
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

        # 畫 Rrup and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 2],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='original value')  #數據點
        plt.scatter(x_test[:,2], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'predicted value (accuracy: %.2f)' % (score)) #迴歸線
        plt.xlabel('ln(Rrup)(km)')
        plt.ylabel(f'Predicted ln({self.target})(cm/s^2)')
        plt.title(f'{self.model_name} Predicted Distribution')
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Predict.jpg',
            dpi=300)
        plt.show()

    def residual(self, x_total: "ori_feature", y_total: "ori_ans",
                 predict_value: "ori_predicted",
                 ori_full_data: "ori_notsplit_data", score):
        """
        
        Total Residual 
        
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

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(np.exp(x_total[:, 0])[i], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(np.exp(x_total[:, 0])[j], 2) - 1e2) / (
                (2 * 1e3 - 1e2) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(np.exp(x_total[:, 0]),
                    residual,
                    c=color_column,
                    cmap=newcmp)
        plt.colorbar(extend='both', label='number value')
        plt.scatter([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean,
            residual_794_mean, residual_1000_mean
        ],
                    marker='o',
                    color='black',
                    label='mean value')
        plt.plot([121, 121], [
            residual_121_mean + residual_121_std,
            residual_121_mean - residual_121_std
        ],
                 'black',
                 label='1 std.')
        plt.plot([199, 199], [
            residual_199_mean + residual_199_std,
            residual_199_mean - residual_199_std
        ], 'black')
        plt.plot([398, 398], [
            residual_398_mean + residual_398_std,
            residual_398_mean - residual_398_std
        ], 'black')
        plt.plot([794, 794], [
            residual_794_mean + residual_794_std,
            residual_794_mean - residual_794_std
        ], 'black')
        plt.plot([1000, 1000], [
            residual_1000_mean + residual_1000_std,
            residual_1000_mean - residual_1000_std
        ], 'black')

        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean,
            residual_794_mean, residual_1000_mean
        ], 'k--')
        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean + residual_121_std, residual_199_mean +
            residual_199_std, residual_398_mean + residual_398_std,
            residual_794_mean + residual_794_std,
            residual_1000_mean + residual_1000_std
        ], 'k--')
        plt.plot([121, 199, 398, 794, 1000], [
            residual_121_mean - residual_121_std, residual_199_mean -
            residual_199_std, residual_398_mean - residual_398_std,
            residual_794_mean - residual_794_std,
            residual_1000_mean - residual_1000_std
        ], 'k--')
        plt.xscale("log")
        plt.xlim(1e2, 2 * 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Vs30(m/s)')
        plt.legend()
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
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

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(x_total[:, 1][i], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(x_total[:, 1][j], 2) - 3) / ((8 - 3) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(x_total[:, 1], residual, c=color_column, cmap=newcmp)
        plt.colorbar(extend='both', label='number value')
        plt.scatter([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean,
            residual_6_5_mean
        ],
                    marker='o',
                    color='black',
                    label='mean value')
        plt.plot([3.5, 3.5], [
            residual_3_5_mean + residual_3_5_std,
            residual_3_5_mean - residual_3_5_std
        ],
                 'black',
                 label='1 std.')
        plt.plot([4.5, 4.5], [
            residual_4_5_mean + residual_4_5_std,
            residual_4_5_mean - residual_4_5_std
        ], 'black')
        plt.plot([5.5, 5.5], [
            residual_5_5_mean + residual_5_5_std,
            residual_5_5_mean - residual_5_5_std
        ], 'black')
        plt.plot([6.5, 6.5], [
            residual_6_5_mean + residual_6_5_std,
            residual_6_5_mean - residual_6_5_std
        ], 'black')

        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean,
            residual_6_5_mean
        ], 'k--')
        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean + residual_3_5_std, residual_4_5_mean +
            residual_4_5_std, residual_5_5_mean + residual_5_5_std,
            residual_6_5_mean + residual_6_5_std
        ], 'k--')
        plt.plot([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean - residual_3_5_std, residual_4_5_mean -
            residual_4_5_std, residual_5_5_mean - residual_5_5_std,
            residual_6_5_mean - residual_6_5_std
        ], 'k--')
        plt.xlim(3, 8)
        plt.ylim(-3, 3)
        plt.xlabel('Mw')
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.legend()
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
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

        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(np.exp(x_total[:, 2])[i], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(residual[i], 2) - (-3)) / ((3 - (-3)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 資料落在第x,y個網格就+1紀錄
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(np.exp(x_total[:, 2])[j], 2) - 5 * 1e0) / (
                (1e3 - 5 * 1e0) / net)
            y_net = (round(residual[j], 2) - (-3)) / ((3 - (-3)) / net)
            color_column.append(zz[math.floor(x_net),
                                   math.floor(y_net)])  #將統計好資料寫入color_column
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(np.exp(x_total[:, 2]),
                    residual,
                    c=color_column,
                    cmap=newcmp)
        plt.colorbar(extend='both', label='number value')
        plt.scatter([10, 31, 100, 316], [
            residual_10_mean, residual_31_mean, residual_100_mean,
            residual_316_mean
        ],
                    marker='o',
                    color='black',
                    label='mean value')
        plt.plot([10, 10], [
            residual_10_mean + residual_10_std,
            residual_10_mean - residual_10_std
        ],
                 'black',
                 label='1 std.')
        plt.plot([31, 31], [
            residual_31_mean + residual_31_std,
            residual_31_mean - residual_31_std
        ], 'black')
        plt.plot([100, 100], [
            residual_100_mean + residual_100_std,
            residual_100_mean - residual_100_std
        ], 'black')
        plt.plot([316, 316], [
            residual_316_mean + residual_316_std,
            residual_316_mean - residual_316_std
        ], 'black')

        plt.plot([10, 31, 100, 316], [
            residual_10_mean, residual_31_mean, residual_100_mean,
            residual_316_mean
        ], 'k--')
        plt.plot([10, 31, 100, 316], [
            residual_10_mean + residual_10_std, residual_31_mean +
            residual_31_std, residual_100_mean + residual_100_std,
            residual_316_mean + residual_316_std
        ], 'k--')
        plt.plot([10, 31, 100, 316], [
            residual_10_mean - residual_10_std, residual_31_mean -
            residual_31_std, residual_100_mean - residual_100_std,
            residual_316_mean - residual_316_std
        ], 'k--')
        plt.xscale("log")
        plt.xlim(5 * 1e0, 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Rrup(km)')
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.legend()
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
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
        plt.bar(x_bar, total_num_residual, edgecolor='white', width=0.2)

        mu = np.mean(residual)  # mean and standard deviation 可以改，如果分布有變的話
        sigma = np.std(residual)
        plt.text(2, 2700, f'mean = {round(mu,2)}')
        plt.text(2, 1700, f'sd = {round(sigma,2)}')
        plt.grid(linestyle=':', color='darkgrey')
        plt.xlabel('Total-Residual')
        plt.ylabel('Numbers')
        plt.title(f'{self.abbreviation_name} Total-Residual Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Total-Residual Distribution.jpg',
            dpi=300)
        plt.show()
        """

        inter intra event residual
        
        """

        # 計算inter-event by mean value(共273顆地震)
        originaldata_predicted_result_df = pd.DataFrame(predict_value,
                                                        columns=['predicted'])
        total_data_df = pd.concat(
            [ori_full_data, originaldata_predicted_result_df], axis=1)
        # 這裡看是否需要變log10
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
        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(inter_event['Mw'],
                    inter_event['inter_event_residual'],
                    marker='o',
                    s=8,
                    facecolors='None',
                    edgecolors='black')
        plt.plot([3, 8], [
            inter_event['inter_event_residual'].mean(),
            inter_event['inter_event_residual'].mean()
        ],
                 'b--',
                 linewidth=0.5)
        plt.xlim(3, 8)
        plt.ylim(-1.6, 1.6)
        plt.yticks(Mw_yticks)
        plt.xlabel('Mw')
        plt.ylabel(f'Inter-event Residual ln({self.target})(cm/s^2)')
        inter_mean = round(inter_event['inter_event_residual'].mean(), 2)
        plt.title(
            f'{self.abbreviation_name} Inter-event Residual Mean:{inter_mean}')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Inter-event Residual.jpg',
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
        Rrup_xticks = [0, 50, 100, 150, 200, 250, 300, 400, 500]
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(total_data_df['Rrup'][i], 2) -
                     (-50)) / ((600 - (-50)) / net) # -50為圖中最小值 600為圖中最大值
            y_net = (round(total_data_df['intra_event_residual'][i], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net) # -2.5為圖中最小值 2.5為圖中最大值
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(total_data_df['Rrup'][j], 2) -
                     (-50)) / ((600 - (-50)) / net)
            y_net = (round(total_data_df['intra_event_residual'][j], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(total_data_df['Rrup'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    c=color_column,
                    cmap=newcmp)
        plt.colorbar(extend='both', label='number value')
        plt.plot([-50, 600], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
                 'b--',
                 label="mean value",
                 linewidth=0.5)
        plt.xlim(-50, 600)
        plt.ylim(-2.5, 2.5)
        plt.xticks(Rrup_xticks)
        plt.legend()
        plt.xlabel('Rrup(km)')
        plt.ylabel(f'Intra-event Residual ln({self.target})(cm/s^2)')
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean}')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Intra-event Residual.png',
            dpi=300)
        plt.show()

        # Vs30 intra-event
        Vs30_xticks = [200, 400, 600, 800, 1000, 1200, 1400]
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(residual):  # 計算每個網格中總點數
            x_net = (round(total_data_df['Rrup'][i], 2) - 0) / (
                (1400 - 0) / net)
            y_net = (round(total_data_df['intra_event_residual'][i], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
            i += 1

        j = 0
        while j < len(residual):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(total_data_df['Rrup'][j], 2) - 0) / (
                (1400 - 0) / net)
            y_net = (round(total_data_df['intra_event_residual'][j], 2) -
                     (-2.5)) / ((2.5 - (-2.5)) / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(total_data_df['Vs30'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    c=color_column,
                    cmap=newcmp)
        plt.colorbar(extend='both', label='number value')
        plt.plot([0, 1400], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
                 'b--',
                 label="mean value",
                 linewidth=0.5)
        plt.xlim(0, 1400)
        plt.ylim(-2.5, 2.5)
        plt.xticks(Vs30_xticks)
        plt.legend()
        plt.xlabel('Vs30(m/s)')
        plt.ylabel(f'Intra-event Residual ln({self.target})(cm/s^2)')
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean}')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Intra-event Residual.jpg',
            dpi=300)
        plt.show()

    def measured_predict(self, y_test: "ori_ans",
                         predict_value: "ori_predicted", score):
        """
        
        預測PGA和實際PGA
        
        """
        net = 50
        zz = np.array([0] * net * net).reshape(net, net)  # 打net*net個網格
        color_column = []

        i = 0
        while i < len(y_test):  # 計算每個網格中總點數
            x_net = (round(y_test[i], 2) + 2) / (10 / net)
            # +2:因為網格從-2開始打 10:頭減尾8-(-2) 10/net:網格間格距離 x_net:x方向第幾個網格
            y_net = (round(predict_value[i], 2) + 2) / (10 / net)
            zz[math.floor(x_net), math.floor(y_net)] += 1  # 第x,y個網格
            i += 1

        j = 0
        while j < len(y_test):  # 並非所有網格都有用到，沒用到的就不要畫進圖裡
            x_net = (round(y_test[j], 2) + 2) / (10 / net)
            y_net = (round(predict_value[j], 2) + 2) / (10 / net)
            color_column.append(zz[math.floor(x_net), math.floor(y_net)])
            # color_column:依照資料落在哪個網格給定該資料顏色值
            j += 1

        colorlist = ["darkgrey", "blue", "yellow", "orange", "red"]
        newcmp = LinearSegmentedColormap.from_list('testCmap',
                                                   colors=colorlist,
                                                   N=256)

        plt.grid(linestyle=':')
        plt.scatter(y_test, predict_value, c=color_column, cmap=newcmp)
        x_line = [-2, 8]
        y_line = [-2, 8]
        plt.plot(x_line, y_line, 'r--', alpha=0.5)
        plt.xlabel(f'Measured ln({self.target})(cm/s^2)')
        plt.ylabel(f'Predict ln({self.target})(cm/s^2)')
        plt.ylim(-2, 8)
        plt.xlim(-2, 8)
        plt.title(
            f'{self.SMOGN_TSMIP} {self.abbreviation_name} Measured Predicted Distribution'
        )
        plt.text(4.5, 0, f"R2 score = {round(score,2)}")
        plt.colorbar(extend='both', label='number value')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Measured Predicted Comparison.png',
            dpi=300)
        plt.show()

    def distance_scaling(
            self,  # change Mw Vs30 etc. condition by csv file
            Vs30,
            Mw,
            Rrup,
            fault_type,
            station_rank,
            x_total: "ori_feature",
            y_total: "ori_ans",
            ML_model):
        DSCon = np.array(
            [[np.log(Vs30), Mw,
              np.log(Rrup), fault_type, station_rank]])
        Result = []
        for i in np.linspace(1, 200, 100):
            DSCon[0][2] = np.log(round(i, 2))  # 給定距離預測值
            Result.append(np.exp(ML_model.predict(DSCon)) / 980)
        myline = np.linspace(1, 200, 100)
        fault_type_list = ["0", "REV", "NM", "SS"]
        x_total = np.transpose(x_total, (1, 0))  # 轉換dim

        fig = plt.figure()
        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.plot(myline,
                 Result,
                 linewidth='0.8',
                 color='r',
                 label=fault_type_list[fault_type])
        plt.scatter(np.exp(x_total[2]),
                    np.exp(y_total)/980,
                    marker='o',
                    facecolors='none', 
                    color='grey',
                    label='data')
        plt.xlabel('Rrup(km)')
        plt.ylabel(f"PSA({self.target})(g)")
        plt.ylim(10e-5, 10)
        plt.xlim(1, 300)
        plt.yscale("log")
        plt.xscale("log")
        plt.xticks([1, 10, 50, 100, 200, 300], [1, 10, 50, 100, 200, 300])
        plt.title(f"M = {Mw}, Vs30 = {Vs30}m/s")
        plt.legend()
        plt.savefig(
            f"distance scaling-{self.target} Mw{Mw} Vs30{Vs30} fault-type{fault_type} station{station_rank}.jpg",
            dpi=300)
        plt.show()

    def explainable(self, x_test: "ori_test_feature", model_feture, ML_model,
                    seed):
        df = pd.DataFrame(x_test, columns=model_feture)
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
        plt.savefig(f"shap_scatter_MW_lnRrup_{self.target}.jpg",
                    bbox_inches='tight',
                    dpi=300)

        fig = plt.figure()
        shap.plots.scatter(shap_values[:, "lnRrup"],
                           color=shap_values[:, "MW"],
                           show=False)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.savefig(f"shap_scatter_lnRrup_MW_{self.target}.jpg",
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

    def respond_spetrum(self, Vs30, Mw, Rrup, fault_type, station_rank,
                        *args: "model"):
        RSCon = np.array(
            [[np.log(Vs30), Mw,
              np.log(Rrup), fault_type, station_rank]])
        PGA_model = pickle.load(open(args[0], 'rb'))
        PGV_model = pickle.load(open(args[1], 'rb'))
        Sa001_model = pickle.load(open(args[2], 'rb'))
        Sa005_model = pickle.load(open(args[3], 'rb'))
        Sa01_model = pickle.load(open(args[4], 'rb'))
        Sa02_model = pickle.load(open(args[5], 'rb'))
        Sa05_model = pickle.load(open(args[6], 'rb'))
        Sa10_model = pickle.load(open(args[7], 'rb'))
        Sa30_model = pickle.load(open(args[8], 'rb'))
        Sa40_model = pickle.load(open(args[9], 'rb'))
        Sa100_model = pickle.load(open(args[10], 'rb'))

        PGA_predict = np.exp(PGA_model.predict(RSCon)) / 980
        PGV_predict = np.exp(PGV_model.predict(RSCon)) / 980
        Sa001_predict = np.exp(Sa001_model.predict(RSCon)) / 980
        Sa005_predict = np.exp(Sa005_model.predict(RSCon)) / 980
        Sa01_predict = np.exp(Sa01_model.predict(RSCon)) / 980
        Sa02_predict = np.exp(Sa02_model.predict(RSCon)) / 980
        Sa05_predict = np.exp(Sa05_model.predict(RSCon)) / 980
        Sa10_predict = np.exp(Sa10_model.predict(RSCon)) / 980
        Sa30_predict = np.exp(Sa30_model.predict(RSCon)) / 980
        Sa40_predict = np.exp(Sa40_model.predict(RSCon)) / 980
        Sa100_predict = np.exp(Sa100_model.predict(RSCon)) / 980
        fault_type_list = ["0", "REV", "NM", "SS"]
        plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5)
        plt.plot([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 10.0], [
            Sa001_predict[0], Sa005_predict[0], Sa01_predict[0],
            Sa02_predict[0], Sa05_predict[0], Sa10_predict[0], Sa30_predict[0],
            Sa40_predict[0], Sa100_predict[0]
        ],
                 label=fault_type_list[fault_type])
        plt.title(f"M = {Mw}, Rrup = {Rrup}km, Vs30 = {Vs30}m/s")
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
            f"response spectrum-Mw{Mw} Rrup{Rrup} Vs30{Vs30} fault-type{fault_type} station{station_rank}.jpg",
            dpi=300)


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
