import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
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

        #######################################
        #######################################
        #######################################
        ###### train & test distribution ######
        #######################################
        #######################################
        #######################################

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

    def residual(self, x_total, y_total, predict_value, ori_full_data, score):

        ##########################
        ##### Total Residual #####
        ##########################

        # 1. Vs30 Total Residual
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
        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(np.exp(x_total[:, 0]),
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean,
            residual_794_mean, residual_1000_mean
        ],
                    marker='o',
                    facecolors='none',
                    edgecolors='b')
        plt.plot([121, 121], [
            residual_121_mean + residual_121_std,
            residual_121_mean - residual_121_std
        ], 'b')
        plt.plot([199, 199], [
            residual_199_mean + residual_199_std,
            residual_199_mean - residual_199_std
        ], 'b')
        plt.plot([398, 398], [
            residual_398_mean + residual_398_std,
            residual_398_mean - residual_398_std
        ], 'b')
        plt.plot([794, 794], [
            residual_794_mean + residual_794_std,
            residual_794_mean - residual_794_std
        ], 'b')
        plt.plot([1000, 1000], [
            residual_1000_mean + residual_1000_std,
            residual_1000_mean - residual_1000_std
        ], 'b')
        plt.xscale("log")
        plt.xlim(1e2, 2 * 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Vs30(m/s)')
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict Residual.jpg',
            dpi=300)
        plt.show()

        # 2. Mw Toral Residual
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

        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(x_total[:, 1],
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean,
            residual_6_5_mean
        ],
                    marker='o',
                    facecolors='none',
                    edgecolors='b')
        plt.plot([3.5, 3.5], [
            residual_3_5_mean + residual_3_5_std,
            residual_3_5_mean - residual_3_5_std
        ], 'b')
        plt.plot([4.5, 4.5], [
            residual_4_5_mean + residual_4_5_std,
            residual_4_5_mean - residual_4_5_std
        ], 'b')
        plt.plot([5.5, 5.5], [
            residual_5_5_mean + residual_5_5_std,
            residual_5_5_mean - residual_5_5_std
        ], 'b')
        plt.plot([6.5, 6.5], [
            residual_6_5_mean + residual_6_5_std,
            residual_6_5_mean - residual_6_5_std
        ], 'b')
        plt.xlim(3, 8)
        plt.ylim(-3, 3)
        plt.xlabel('Mw')
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict Residual.jpg',
            dpi=300)
        plt.show()

        # 3. Rrup Total Residual
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
        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(np.exp(x_total[:, 2]),
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter([10, 31, 100, 316], [
            residual_10_mean, residual_31_mean, residual_100_mean,
            residual_316_mean
        ],
                    marker='o',
                    facecolors='none',
                    edgecolors='b')
        plt.plot([10, 10], [
            residual_10_mean + residual_10_std,
            residual_10_mean - residual_10_std
        ], 'b')
        plt.plot([31, 31], [
            residual_31_mean + residual_31_std,
            residual_31_mean - residual_31_std
        ], 'b')
        plt.plot([100, 100], [
            residual_100_mean + residual_100_std,
            residual_100_mean - residual_100_std
        ], 'b')
        plt.plot([316, 316], [
            residual_316_mean + residual_316_std,
            residual_316_mean - residual_316_std
        ], 'b')
        plt.xscale("log")
        plt.xlim(5 * 1e0, 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Rrup(km)')
        plt.ylabel(f'Residual ln({self.target})(cm/s^2)')
        plt.title(
            f'{self.abbreviation_name} Predicted Residual R2 score: %.3f' %
            (score))
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Predict Residual.jpg',
            dpi=300)
        plt.show()

        # 4. Total Residual Number Distribution
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
        x_nor = np.linspace(-4, 4, 100)
        plt.plot(x_nor, (1 / (sigma * np.sqrt(2 * np.pi)) *
                         np.exp(-(x_nor - mu)**2 / (2 * sigma**2)) * 11000),
                 linewidth=1,
                 color='r')
        plt.grid(linestyle=':', color='darkgrey')
        plt.xlabel('Total-Residual')
        plt.ylabel('Numbers')
        plt.title(f'{self.abbreviation_name} Total-Residual Distribution')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Total-Residual Distribution.jpg',
            dpi=300)
        plt.show()

        #######################################
        #######################################
        #######################################
        ##### inter intra event residual  #####
        #######################################
        #######################################
        #######################################

        # 計算inter-event by mean value(共273顆地震)
        originaldata_predicted_result_df = pd.DataFrame(predict_value,
                                                        columns=['predicted'])
        total_data_df = pd.concat(
            [ori_full_data, originaldata_predicted_result_df], axis=1)
        # 這裡看是否需要變log10
        # total_data_df["residual"] = np.abs((np.exp(total_data_df["predicted"]) - np.exp(total_data_df["lnPGA(gal)"]))/980)
        total_data_df["residual"] = total_data_df["predicted"] - total_data_df[
            "lnPGA(gal)"]
        
        # build new dataframe to collect inter-event value
        summeries = {'residual': 'mean', 'MW': 'max'}
        inter_event = total_data_df.groupby(
            by="EQ_ID").agg(summeries).reset_index()
        inter_event = inter_event.rename(columns={
            'residual': 'inter_event_residual',
            'MW': 'Mw'
        })

        # Mw inter-event
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

        # Rrup intra-event
        Rrup_xticks = [0, 50, 100, 150, 200, 250, 300, 400, 500]
        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(total_data_df['Rrup'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    facecolors='None',
                    edgecolors='black')
        plt.plot([-50, 600], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
                 'b--',
                 linewidth=0.5)
        plt.xlim(-50, 600)
        plt.xticks(Rrup_xticks)
        plt.xlabel('Rrup(km)')
        plt.ylabel(f'Intra-event Residual ln({self.target})(cm/s^2)')
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean}')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Intra-event Residual.jpg',
            dpi=300)
        plt.show()

        # Vs30 intra-event
        Vs30_xticks = [200, 400, 600, 800, 1000, 1200, 1400]
        plt.grid(linestyle=':', color='darkgrey')
        plt.scatter(total_data_df['Vs30'],
                    total_data_df['intra_event_residual'],
                    marker='o',
                    s=8,
                    facecolors='None',
                    edgecolors='black')  #迴歸線
        plt.plot([0, 1400], [
            total_data_df['intra_event_residual'].mean(),
            total_data_df['intra_event_residual'].mean()
        ],
                 'b--',
                 linewidth=0.5)
        plt.xlim(0, 1400)
        plt.xticks(Vs30_xticks)
        plt.xlabel('Vs30(m/s)')
        plt.ylabel(f'Intra-event Residual ln({self.target})(cm/s^2)')
        intra_mean = round(total_data_df['intra_event_residual'].mean(), 2)
        plt.title(
            f'{self.abbreviation_name} Intra-event Residual Mean:{intra_mean}')
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Intra-event Residual.jpg',
            dpi=300)
        plt.show()

    def measured_predict(self, y_test, predict_value, score):

        #./
        # 預測PGA和實際PGA
        # /.

        plt.grid(linestyle=':')
        plt.scatter(y_test, predict_value,marker='o',facecolors='none',edgecolors='r', \
            label='Data') #迴歸線.
        x_line = [-5, 10]
        y_line = [-5, 10]
        plt.plot(x_line, y_line, color='blue')
        plt.xlabel(f'Measured ln({self.target})(cm/s^2)')
        plt.ylabel(f'Predict ln({self.target})(cm/s^2)')
        plt.ylim(-2, 8)
        plt.xlim(-2, 8)
        plt.title(
            f'{self.SMOGN_TSMIP} {self.abbreviation_name} Measured Predicted Distribution'
        )
        plt.text(5, 0, f"R2 score = {round(score,2)}")
        plt.legend()
        plt.savefig(
            f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} {self.target} {self.abbreviation_name} Measured Predicted Comparison.jpg',
            dpi=300)
        plt.show()

    def distance_scaling(self,
                         original_data_feature,
                         originaldata_predicted_result,
                         minVs30,
                         maxVs30,
                         minMw,
                         maxMw,
                         faulttype,
                         score,
                         poly=None):

        # ./
        # PGA隨距離的衰減
        # /.

        concate_predicted = np.concatenate(
            (original_data_feature[0], originaldata_predicted_result[:, None]),
            axis=1)  # 預測後PGA & 原本資料特徵檔合併
        predicted_df = pd.DataFrame(concate_predicted,
                                    columns=[
                                        'lnVs30', 'MW', 'lnRrup', 'fault.type',
                                        'STA_Lon_X', 'STA_Lat_Y',
                                        f'predicted_ln{self.target}(gal)'
                                    ])  # ndarray 轉 dataframe，方便後續處理

        # ./
        # Poly回歸線
        # /.
        plt.scatter(predicted_df['lnRrup'][predicted_df['MW'] >= minMw][
            predicted_df['MW'] < maxMw][
                predicted_df['fault.type'] == faulttype][
                    np.exp(predicted_df['lnVs30']) > minVs30][
                        np.exp(predicted_df['lnVs30']) < maxVs30],
                    predicted_df[f'predicted_ln{self.target}(gal)']
                    [predicted_df['MW'] >= minMw][predicted_df['MW'] < maxMw][
                        predicted_df['fault.type'] == faulttype][
                            np.exp(predicted_df['lnVs30']) > minVs30][
                                np.exp(predicted_df['lnVs30']) < maxVs30],
                    label=f"Mw{round((maxMw + minMw)/2,1)}")
        # fit by polynomial
        mymodel = np.poly1d(
            np.polyfit(
                predicted_df['lnRrup'][predicted_df['MW'] >= minMw]
                [predicted_df['MW'] < maxMw][
                    predicted_df['fault.type'] == faulttype][
                        np.exp(predicted_df['lnVs30']) > minVs30][
                            np.exp(predicted_df['lnVs30']) < maxVs30].values,
                predicted_df[f'predicted_ln{self.target}(gal)'][
                    predicted_df['MW'] >= minMw][predicted_df['MW'] < maxMw][
                        predicted_df['fault.type'] == faulttype]
                [np.exp(predicted_df['lnVs30']) > minVs30][
                    np.exp(predicted_df['lnVs30']) < maxVs30].values, poly))
        myline = np.linspace(1, 6, 50)
        plt.grid(linestyle=':')
        plt.plot(myline,
                 mymodel(myline),
                 linewidth='0.8',
                 color='r',
                 label=f"polynomial line degree{poly}")

        # ./
        # 取平均法
        # /.

        # predicted_df['average_lnRrup'] = 0
        # predicted_df[f'average_predicted_ln{self.target}(gal)'] = 0

        # plt.scatter(predicted_df['lnRrup'][predicted_df['MW'] >= minMw][
        #     predicted_df['MW'] < maxMw][
        #         predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #             predicted_df[f'predicted_ln{self.target}(gal)']
        #             [predicted_df['MW'] >= minMw][predicted_df['MW'] < maxMw][
        #                 predicted_df['fault.type'] == faulttype][
        #                     np.exp(predicted_df['lnVs30']) > minVs30][
        #                         np.exp(predicted_df['lnVs30']) < maxVs30],
        #             label=f"Mw{round((maxMw + minMw)/2,1)}")

        # # 計算平均
        # predicted_df = predicted_df.sort_values(by=['lnRrup'])
        # for i in range(0, len(predicted_df['lnRrup']) - 1):
        #     predicted_df['average_lnRrup'][i] = (predicted_df['lnRrup'][i] +
        #                               predicted_df['lnRrup'][i + 1]) / 2
        #     predicted_df[f'average_predicted_ln{self.target}(gal)'][i] = (
        #         predicted_df[f'predicted_ln{self.target}(gal)'][i] +
        #         predicted_df[f'predicted_ln{self.target}(gal)'][i + 1]) / 2
        # predicted_df = predicted_df.sort_values(by=['average_lnRrup'])

        # plt.grid(linestyle=':')
        # plt.plot(predicted_df['average_lnRrup'][predicted_df['MW'] >= minMw][
        #     predicted_df['MW'] < maxMw][
        #         predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #          predicted_df[f'average_predicted_ln{self.target}(gal)'][
        #              predicted_df['MW'] >= minMw][predicted_df['MW'] < maxMw][
        #                  predicted_df['fault.type'] == faulttype][
        #                      np.exp(predicted_df['lnVs30']) > minVs30][
        #                          np.exp(predicted_df['lnVs30']) < maxVs30],
        #          linewidth='0.8',
        #          color='r',
        #          label=f"Mw{round((maxMw + minMw)/2,1)}")

        # ./
        # plot line
        # /.

        # plt.plot(
        #     predicted_df['lnRrup'][predicted_df['MW'] >= minMw][
        #         predicted_df['MW'] < maxMw][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     predicted_df[f'predicted_ln{self.target}(gal)'][predicted_df['MW'] >= minMw][
        #         predicted_df['MW'] < maxMw][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     linewidth='0.8',
        #     label=f"Mw{round((maxMw + minMw)/2,1)}")

        # ./
        # plot total line
        # /.

        # plt.grid(linestyle=':')
        # plt.plot(
        #     predicted_df['lnRrup'][predicted_df['MW'] >= 4.0][
        #         predicted_df['MW'] < 5.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     predicted_df[f'predicted_ln{self.target}(gal)'][predicted_df['MW'] >=4.0][
        #         predicted_df['MW'] < 5.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     linewidth='0.8',
        #     label="Mw4.5")
        # plt.plot(
        #     predicted_df['lnRrup'][predicted_df['MW'] >= 5.0][
        #         predicted_df['MW'] < 6.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     predicted_df[f'predicted_ln{self.target}(gal)'][predicted_df['MW'] >= 5.0][
        #         predicted_df['MW'] < 6.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     linewidth='0.8',
        #     label="Mw5.5")
        # plt.plot(
        #     predicted_df['lnRrup'][predicted_df['MW'] >= 6.0][
        #         predicted_df['MW'] < 7.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     predicted_df[f'predicted_ln{self.target}(gal)'][predicted_df['MW'] >= 6.0][
        #         predicted_df['MW'] < 7.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     linewidth='0.8',
        #     label="Mw6.5")
        # plt.plot(
        #     predicted_df['lnRrup'][predicted_df['MW'] >= 7.0][
        #         predicted_df['MW'] < 8.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     predicted_df['predicted_lnPGA(gal)'][predicted_df['MW'] >= 7.0][
        #         predicted_df['MW'] < 8.0][predicted_df['fault.type'] == faulttype][
        #             np.exp(predicted_df['lnVs30']) > minVs30][
        #                 np.exp(predicted_df['lnVs30']) < maxVs30],
        #     linewidth='0.8',
        #     label="Mw7.5")

        plt.xlabel('ln(Rrup)(km)')
        plt.ylabel(f'Predicted ln({self.target})(cm/s^2)')
        plt.title(f'{self.abbreviation_name} Distance Scaling')

        if self.target == "PGA":
            plt.text(0.5, 3, f"R2 score = {round(score,2)}")
            if faulttype == 1:
                plt.text(0.5, 2, f"Reverse")
            elif faulttype == 2:
                plt.text(0.5, 2, f"Normal")
            elif faulttype == 3:
                plt.text(0.5, 2, f"Strike Slip")
            plt.text(0.5, 1, f"{minVs30}<Vs30<{maxVs30}")
        elif self.target == "PGV":
            plt.text(1, 6.5, f"R2 score = {round(score,2)}")
            if faulttype == 1:
                plt.text(1, 5.5, f"Reverse")
            elif faulttype == 2:
                plt.text(1, 5.5, f"Normal")
            elif faulttype == 3:
                plt.text(1, 5.5, f"Strike Slip")
            plt.text(1, 4.5, f"{minVs30}<Vs30<{maxVs30}")

        if self.target == "PGA":
            plt.xlim(0, 6)
            plt.ylim(0.4, 7)
        elif self.target == "PGV":
            plt.xlim(0.5, 6)
        plt.legend()
        # plt.savefig(
        #     f'../{self.abbreviation_name}/{f"{self.target} Mw{round((maxMw + minMw)/2,1)} faulttype={faulttype} {minVs30}_Vs30_{maxVs30}"} Distance Scaling.jpg',
        #     dpi=300)
        plt.savefig(
            f'../{self.abbreviation_name}/{f"{self.target} Mw{round((maxMw + minMw)/2,1)} faulttype={faulttype} {minVs30}_Vs30_{maxVs30}"} Distance Scaling Continue line degree{poly}.jpg',
            dpi=300)
        plt.show()

    def inter_intra_event(self):
        print("hi")


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
