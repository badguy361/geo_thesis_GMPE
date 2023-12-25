import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from process_train import dataprocess
import matplotlib.pyplot as plt
import smogn


def get_sta_rank():
    """

    計算station rank，設定一基準點，並計算各測站與其距離

    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")
    point = [119.5635611, 21.90093889]  # 最左下角為基準
    STA_dist = (((TSMIP_df["STA_Lat_Y"] - point[1]) * 110)**2 +
                ((TSMIP_df["STA_Lon_X"] - point[0]) * 101)**2)**(1 / 2)
    TSMIP_df["STA_DIST"] = STA_dist
    TSMIP_df["STA_rank"] = TSMIP_df["STA_DIST"].rank(method='dense')

    for i in range(1, 38):
        plt.figure(figsize=(20, 6))
        plt.bar(
            list(dict(Counter(TSMIP_df["STA_ID"])).keys())[
                20 * (i - 1):20 * i],
            list(dict(Counter(TSMIP_df["STA_ID"])).values())[20 * (i - 1):20 * i])
        plt.title(f'STA Distribution ID_{20*(i-1)}-{20*i}')
        plt.savefig(f"STA_ID_distribution_{20*(i-1)}-{20*i}.jpg", dpi=300)
    # len(TSMIP_df["STA_ID"].unique())
    plt.scatter(TSMIP_df["STA_Lon_X"],
                TSMIP_df["STA_Lat_Y"],
                c=TSMIP_df["STA_rank"],
                cmap='bwr')
    plt.colorbar(extend='both', label='number value')
    plt.plot(point[0], point[1], "*", color='black')
    plt.title('TSMIP Station located')
    plt.savefig("../XGB/STA_ID.jpg", dpi=300)
    plt.show()


def cut_period(target, period, column):
    """

    Args:
        target ([str]): [file target name]
        period ([float]): [period which want to be filterd]
        column ([str]): [column name in the FF]

    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")

    filtered_df = TSMIP_df[TSMIP_df["usable_period_H"] > period]
    selected_columns_df = filtered_df[["EQ_ID", "MW", "fault.type", "Vs30", "Rrup",
                                       "STA_Lon_X", "STA_Lat_Y", "STA_rank", "STA_ID", "eq.type", "Z1.0", f"{column}"]]

    selected_columns_df.to_csv(
        f"../../../TSMIP_FF_period/TSMIP_FF_{target}.csv", index=False)


def pre_SMOGN(target, column):
    """
    
    process the pre_SMOGN for a particular target .

    Args:
        target ([str]): [file target name]
        column ([str]): [column name in the FF]
    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_period/TSMIP_FF_{target}.csv")
    filtered_df = TSMIP_df[TSMIP_df["eq.type"] == "shallow crustal"]
    selected_columns_df = filtered_df[["MW", "fault.type", "Vs30", "Rrup",
                                    "STA_rank", f"{column}"]]
    
    selected_columns_df.to_csv(
        f"../../../TSMIP_FF_pre_SMOGN/TSMIP_FF_pre_smogn_{target}.csv", index=False)


def synthesize(target, column):
    """

    Synthesize dataset through SMOGN method.

    Args:
        target ([str]): [file target name]
        column ([str]): [column name in the FF]
    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_pre_SMOGN/TSMIP_FF_pre_smogn_{target}.csv")

    # conduct smogn ##這步大概要兩個小時
    TSMIP_smogn = smogn.smoter(
        data=TSMIP_df,
        y=column
    )
    TSMIP_smogn.to_csv(f"../../../TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv", index=False)


def SMOGN_plot():
    """

    Plot the figure to compare before SMOGN and after.

    Returns:
        [figure]: [boxplots for original and SMOGN dataset]
    """
    # Compare synthesized data and original data
    target = "PGA"
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")
    TSMIP_smogn_df = pd.read_csv(f"../../../TSMIP_smogn_{target}.csv")
    model = dataprocess()
    after_process_ori_data = model.preProcess(TSMIP_df, target, True)
    after_process_SMOGN_data = model.preProcess(TSMIP_smogn_df, target, False)
    model_feture = ['lnVs30', 'MW', 'lnRrup', 'fault.type', 'STA_rank']
    result_SMOGN = model.splitDataset(after_process_SMOGN_data,
                                      f'ln{target}(gal)', False, *model_feture)
    original_data = model.splitDataset(after_process_ori_data, f'ln{target}(gal)',
                                       False, *model_feture)

    # original dataset box plot
    o0 = np.swapaxes(original_data[0], 0, 1)[0]
    o1 = np.swapaxes(original_data[0], 0, 1)[1]
    o2 = np.swapaxes(original_data[0], 0, 1)[2]
    o3 = np.swapaxes(original_data[0], 0, 1)[3]
    o4 = np.swapaxes(original_data[0], 0, 1)[4]
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 5))
    ax1.boxplot(o0)
    ax1.set_xticklabels(['lnVs30'], fontsize=15)
    ax2.boxplot(o1)
    ax2.set_xticklabels(['Mw'], fontsize=15)
    ax3.boxplot(o2)
    ax3.set_xticklabels(['lnRrup'], fontsize=15)
    ax4.boxplot(o3)
    ax4.set_xticklabels(['rake'], fontsize=15)
    ax4.set_yticks([90, -90, 0])
    ax5.boxplot(o4)
    ax5.set_xticklabels(['station number'], fontsize=15)
    plt.suptitle('Boxplots before SMOGN', fontsize=20)
    plt.savefig('TSMIP boxplots.png', dpi=300)

    # SMOGN box plot
    o0 = np.swapaxes(result_SMOGN[0], 0, 1)[0]
    o1 = np.swapaxes(result_SMOGN[0], 0, 1)[1]
    o2 = np.swapaxes(result_SMOGN[0], 0, 1)[2]
    o3 = np.swapaxes(result_SMOGN[0], 0, 1)[3]
    o4 = np.swapaxes(result_SMOGN[0], 0, 1)[4]
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 5))
    ax1.boxplot(o0)
    ax1.set_xticklabels(['lnVs30'], fontsize=15)
    ax2.boxplot(o1)
    ax2.set_xticklabels(['Mw'], fontsize=15)
    ax3.boxplot(o2)
    ax3.set_xticklabels(['lnRrup'], fontsize=15)
    ax4.boxplot(o3)
    ax4.set_xticklabels(['rake'], fontsize=15)
    ax4.set_yticks([90, -90, 0])
    ax5.boxplot(o4)
    ax5.set_xticklabels(['station number'], fontsize=15)
    plt.suptitle('Boxplots after SMOGN', fontsize=20)
    plt.savefig('TSMIP-SMOGN boxplots.png', dpi=300)
    return 0

targets = ["PGA", "PGV", "Sa001", "Sa005", "Sa01", "Sa02", "Sa03", "Sa04",
           "Sa05", "Sa06", "Sa07", "Sa08", "Sa09", "Sa10", "Sa15", "Sa20",
           "Sa25", "Sa30", "Sa35", "Sa40", "Sa50", "Sa60", "Sa70", "Sa80",
           "Sa90", "Sa100"]
columns = ["PGA", "PGV", "T0.010S", "T0.050S", "T0.100S", "T0.200S", "T0.300S", "T0.400S",
           "T0.500S", "T0.600S", "T0.700S", "T0.800S", "T0.900S", "T1.000S", "T1.500S", "T2.000S",
           "T2.500S", "T3.000S", "T3.500S", "T4.000S", "T5.000S", "T6.000S", "T7.000S", "T8.000S",
           "T9.000S", "T10.000S"]
periods = [0, 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0,
           2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0,9.0, 10.0]

# for i in range(len(targets)):
#     _ = cut_period(targets[i], periods[i], columns[i])

for j in range(len(targets)):
    _ = pre_SMOGN(targets[j], columns[j])

for k in range(len(targets)):
    _ = synthesize(targets[j], columns[j])

# _ = synthesize(target)

# _ = SMOGN_plot()
