import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from design_pattern.process_train import dataprocess

class plot_fig:
    def __init__(self, model_name, abbreviation_name, SMOGN_TSMIP):
        self.model_name = model_name
        self.abbreviation_name = abbreviation_name
        self.SMOGN_TSMIP = SMOGN_TSMIP

    def train_test_distribution(self, x_test, y_test, predict_value, fit_time, score):

    ######################### trainsubset & testsubset distribution #########################

    # 畫 Vs30 and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 0],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='Data')  #數據點
        plt.scatter(x_test[:,0], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'{self.abbreviation_name} (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
        plt.xlabel('Vs30')
        plt.ylabel(f'{self.model_name} Predict')
        plt.title(f'{self.model_name}')
        plt.legend()
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict.png', dpi=300)
        plt.show()

    # 畫 Mw and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 1],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='Data')  #數據點
        plt.scatter(x_test[:,1], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'{self.abbreviation_name} (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
        plt.xlabel('Mw')
        plt.ylabel(f'{self.model_name} Predict')
        plt.title(f'{self.model_name}')
        plt.legend()
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict.png', dpi=300)
        plt.show()

    # 畫 Rrup and randomForest_predict 關係圖
        plt.grid(linestyle=':')
        plt.scatter(x_test[:, 2],
                    y_test,
                    marker='o',
                    facecolors='none',
                    edgecolors='b',
                    label='Data')  #數據點
        plt.scatter(x_test[:,2], predict_value,marker='o',facecolors='none',edgecolors='r', \
            label=f'{self.abbreviation_name} (fit: %.3fs, accuracy: %.3f)' % (fit_time, score)) #迴歸線
        plt.xlabel('lnRrup')
        plt.ylabel(f'{self.model_name} Predict')
        plt.title(f'{self.model_name}')
        plt.legend()
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.model_name} Predict.png', dpi=300)
        plt.show()

    def residual(self, x_test, y_test, predict_value, score):
        
    ######################### residual #########################

    # 1. 計算Vs30_residual
        residual = predict_value - y_test
        residual_121 = []
        residual_199 = []
        residual_398 = []
        residual_794 = []
        residual_1000 = []

        for index, i in enumerate(np.exp(x_test[:, 0])):
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
        plt.scatter(np.exp(x_test[:, 0]),
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter([121, 199, 398, 794, 1000], [
            residual_121_mean, residual_199_mean, residual_398_mean, residual_794_mean,
            residual_1000_mean
        ],
                    marker='o',
                    facecolors='none',
                    edgecolors='b')
        plt.plot([121, 121], [
            residual_121_mean + residual_121_std, residual_121_mean - residual_121_std
        ], 'b')
        plt.plot([199, 199], [
            residual_199_mean + residual_199_std, residual_199_mean - residual_199_std
        ], 'b')
        plt.plot([398, 398], [
            residual_398_mean + residual_398_std, residual_398_mean - residual_398_std
        ], 'b')
        plt.plot([794, 794], [
            residual_794_mean + residual_794_std, residual_794_mean - residual_794_std
        ], 'b')
        plt.plot([1000, 1000], [
            residual_1000_mean + residual_1000_std,
            residual_1000_mean - residual_1000_std
        ], 'b')
        plt.xscale("log")
        plt.xlim(1e2, 2 * 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Vs30(m/s)')
        plt.ylabel('Residual lnPGA(cm/s^2)')
        plt.title(f'{self.SMOGN_TSMIP} {self.abbreviation_name} Predict Residual R2 score: %.3f' % (score))
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Vs30-{self.abbreviation_name} Predict Residual.png', dpi=300)
        plt.show()

    # 2. 計算Mw_residual
        residual = predict_value - y_test
        residual_3_5 = []
        residual_4_5 = []
        residual_5_5 = []
        residual_6_5 = []

        for index, i in enumerate(x_test[:, 1]):
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
        plt.scatter(x_test[:, 1],
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter([3.5, 4.5, 5.5, 6.5], [
            residual_3_5_mean, residual_4_5_mean, residual_5_5_mean, residual_6_5_mean
        ],
                    marker='o',
                    facecolors='none',
                    edgecolors='b')
        plt.plot([3.5, 3.5], [
            residual_3_5_mean + residual_3_5_std, residual_3_5_mean - residual_3_5_std
        ], 'b')
        plt.plot([4.5, 4.5], [
            residual_4_5_mean + residual_4_5_std, residual_4_5_mean - residual_4_5_std
        ], 'b')
        plt.plot([5.5, 5.5], [
            residual_5_5_mean + residual_5_5_std, residual_5_5_mean - residual_5_5_std
        ], 'b')
        plt.plot([6.5, 6.5], [
            residual_6_5_mean + residual_6_5_std, residual_6_5_mean - residual_6_5_std
        ], 'b')
        plt.xlim(3, 8)
        plt.ylim(-3, 3)
        plt.xlabel('Mw')
        plt.ylabel('Residual lnPGA(cm/s^2)')
        plt.title(f'{self.SMOGN_TSMIP} {self.abbreviation_name} Predict Residual R2 score: %.3f' % (score))
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Mw-{self.abbreviation_name} Predict Residual.png', dpi=300)
        plt.show()

    # 3. 計算Rrup_residual
        residual = predict_value - y_test
        residual_10 = []
        residual_31 = []
        residual_100 = []
        residual_316 = []

        for index, i in enumerate(np.exp(x_test[:, 2])):
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
        plt.scatter(np.exp(x_test[:, 2]),
                    residual,
                    marker='o',
                    facecolors='none',
                    edgecolors='r')  #迴歸線
        plt.scatter(
            [10, 31, 100, 316],
            [residual_10_mean, residual_31_mean, residual_100_mean, residual_316_mean],
            marker='o',
            facecolors='none',
            edgecolors='b')
        plt.plot(
            [10, 10],
            [residual_10_mean + residual_10_std, residual_10_mean - residual_10_std],
            'b')
        plt.plot(
            [31, 31],
            [residual_31_mean + residual_31_std, residual_31_mean - residual_31_std],
            'b')
        plt.plot([100, 100], [
            residual_100_mean + residual_100_std, residual_100_mean - residual_100_std
        ], 'b')
        plt.plot([316, 316], [
            residual_316_mean + residual_316_std, residual_316_mean - residual_316_std
        ], 'b')
        plt.xscale("log")
        plt.xlim(5 * 1e0, 1e3)
        plt.ylim(-3, 3)
        plt.xlabel('Rrup(km)')
        plt.ylabel('Residual lnPGA(cm/s^2)')
        plt.title(f'{self.SMOGN_TSMIP} {self.abbreviation_name} Predict Residual R2 score: %.3f' % (score))
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Rrup-{self.abbreviation_name} Predict Residual.png', dpi=300)
        plt.show()

    def measured_predict(self, y_test, predict_value, score):

    ###################### 預測PGA和實際PGA #####################

        plt.grid(linestyle=':')
        plt.scatter(y_test, predict_value,marker='o',facecolors='none',edgecolors='r', \
            label='Data') #迴歸線.
        x_line = [-5, 10]
        y_line = [-5, 10]
        plt.plot(x_line, y_line, color='blue')
        plt.xlabel('Measured PGA')
        plt.ylabel('Predict PGA')
        plt.ylim(-5, 10)
        plt.xlim(-5, 10)
        plt.title(f'{self.SMOGN_TSMIP} {self.abbreviation_name} Measured Predict Distribution')
        plt.text(6, -2, f"R2 score = {round(score,2)}")
        plt.legend()
        plt.savefig(f'../{self.abbreviation_name}/{self.SMOGN_TSMIP} Measured Predict Comparison.png', dpi=300)
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

    plot_something = plot_fig("XGBooster","XGB","TSMIP")
    plot_something.train_test_distribution(result_list[1], result_list[3], final_predict, fit_time, score)
    plot_something.residual(result_list[1], result_list[3], final_predict, score)
    plot_something.measured_predict(result_list[3], final_predict, score)
