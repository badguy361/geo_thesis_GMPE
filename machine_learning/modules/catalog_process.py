import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_train import dataprocess
import matplotlib.pyplot as plt
import smogn
from sklearn.model_selection import train_test_split
import glob

def get_sta_rank():
    """

    計算station rank，設定一基準點，並計算各測站與其距離

    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")
    point = [119.5635611, 21.90093889]  # 最左下角為基準

    #? 算距離
    # STA_dist = (((TSMIP_df["STA_Lat_Y"] - point[1]) * 110)**2 +
    #             ((TSMIP_df["STA_Lon_X"] - point[0]) * 101)**2)**(1 / 2)
    # TSMIP_df["STA_DIST"] = STA_dist
    # TSMIP_df["STA_rank"] = TSMIP_df["STA_DIST"].rank(method='dense')

    #? 測站數分布
    # for i in range(1, 38):
    #     plt.figure(figsize=(20, 6))
    #     plt.bar(
    #         list(dict(Counter(TSMIP_df["STA_ID"])).keys())[
    #             20 * (i - 1):20 * i],
    #         list(dict(Counter(TSMIP_df["STA_ID"])).values())[20 * (i - 1):20 * i])
    #     plt.title(f'STA Distribution ID_{20*(i-1)}-{20*i}')
    #     plt.savefig(f"STA_ID_distribution_{20*(i-1)}-{20*i}.jpg", dpi=300)
    # # len(TSMIP_df["STA_ID"].unique())
    
    #? 測站空間分布
    plt.grid(which="both",
                 axis="both",
                 linestyle="-",
                 linewidth=0.5,
                 alpha=0.5,
                 zorder=0)
    plt.scatter(TSMIP_df["STA_Lon_X"],
                TSMIP_df["STA_Lat_Y"],
                c=TSMIP_df["STA_rank"],
                cmap='bwr',
                s=8,
                zorder=10)
    cbar = plt.colorbar(extend='both', label='number value')
    cbar.set_label('number value', fontsize=12)
    plt.plot(point[0], point[1], "*", color='black')
    plt.xlabel('longitude', fontsize=12)
    plt.ylabel('latitude', fontsize=12)
    plt.title('TSMIP Station id located')
    plt.savefig("../XGB/station_id_distribution.jpg", dpi=300)
    plt.show()

def cut_period(folder, target, period, column):
    """

    Args:
        target ([str]): [file target name]
        period ([float]): [period which want to be filterd]
        column ([str]): [column name in the FF]

    """
    TSMIP_df = pd.read_csv(f"../../../TSMIP_FF.csv")

    filtered_df = TSMIP_df[(TSMIP_df["usable_period_H"] > period) & (TSMIP_df["eq.type"] == "shallow crustal")]
    selected_columns_df = filtered_df[["EQ_ID", "MW", "fault.type", "Vs30", "Rrup",
                                       "STA_Lon_X", "STA_Lat_Y", "STA_rank", "STA_ID", "eq.type", "Z1.0", f"{column}"]]

    selected_columns_df.to_csv(
        f"../../../{folder}/TSMIP_FF_period/TSMIP_FF_{target}.csv", index=False)

def pre_SMOGN(folder, target):
    """
    
    process the pre_SMOGN for a particular target .

    Args:
        target ([str]): [file target name]
    """
    TSMIP_df = pd.read_csv(f"../../../{folder}/TSMIP_FF_period/TSMIP_FF_{target}.csv")
    selected_columns_df = TSMIP_df[["MW", "fault.type", "Vs30", "Rrup",
                                    "STA_rank", f"{target}"]]

    selected_columns_df.to_csv(
        f"../../../{folder}/TSMIP_FF_pre_SMOGN/TSMIP_FF_pre_smogn_{target}.csv", index=False)

def synthesize(folder, target):
    """

    Synthesize dataset through SMOGN method.

    Args:
        target ([str]): [file target name]
    """
    TSMIP_df = pd.read_csv(f"../../../{folder}/TSMIP_FF_pre_SMOGN/TSMIP_FF_pre_smogn_{target}.csv")

    # conduct smogn ##這步大概要兩個小時
    TSMIP_smogn = smogn.smoter(
        data=TSMIP_df,
        y=target
    )
    TSMIP_smogn.to_csv(f"../../../{folder}/TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv", index=False)

def SMOGN_plot(target):
    """

    Plot the figure to compare before SMOGN and after.

    Returns:
        [figure]: [boxplots for original and SMOGN dataset]
    """
    # Compare synthesized data and original data
    TSMIP_df = pd.read_csv(f"../../../cut period_shallow crustal/TSMIP_FF_period/TSMIP_FF_{target}.csv")
    TSMIP_smogn_df = pd.read_csv(f"../../../cut period_shallow crustal/TSMIP_FF_SMOGN/TSMIP_smogn_{target}.csv")
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
    plt.suptitle(f'Boxplots {target} before SMOGN', fontsize=20)
    plt.savefig(f'TSMIP {target} boxplots.png', dpi=300)

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
    plt.suptitle(f'Boxplots {target} after SMOGN', fontsize=20)
    plt.savefig(f'TSMIP-SMOGN {target} boxplots.png', dpi=300)

def period_statistics(periods, *args):
    """
    Statistics the usable periods
    Args:
        periods ([list]): [the period which want to statistics]
        args ([dataframe]): [after SMOGN dataframe whiche need to be calculate total number]
    """
    numbervalue = []
    for data in args[0]:
        numbervalue.append(len(data))

    # 計算長條的左邊和右邊邊界
    left_edges = np.zeros(len(periods))
    right_edges = np.zeros(len(periods))

    left_edges[0] = periods[0] - (periods[1] - periods[0]) / 2
    right_edges[-1] = periods[-1] + (periods[-1] - periods[-2]) / 2
    for i in range(1, len(periods) - 1):
        left_edges[i] = (periods[i] + periods[i - 1]) / 2
        right_edges[i - 1] = left_edges[i]

    # 計算每個長條的寬度
    widths = right_edges - left_edges

    print(numbervalue)
    print(periods)
    plt.bar(periods,numbervalue,width=widths)
    plt.title('Statistic Usable Period')
    plt.xlim(-0.1, 11)
    plt.xscale("symlog")
    plt.xticks([
            0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17,
            0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
            7.5, 10.0]
        , [
            '', '', 0.01, '', '', '', '', '', '', '', '', '', '', '', '', '', 0.5,
            '', '', 1.0, '', '2.0', '', '', 5.0, '', 10.0
        ])
    plt.xlabel('Periods', fontsize=12)
    plt.ylabel('Number of Records', fontsize=12)
    plt.savefig('Statistic Usable Period.png', dpi=300)
    plt.show()

if __name__=='__main__':
    folder = "retry SMOGN"
    #? station id
    # _ = get_sta_rank()

    #? SMOGN
    targets = ["PGA", "PGV", "Sa001", "Sa002", "Sa003", "Sa004", "Sa005",
                "Sa0075", "Sa01", "Sa012", "Sa015", "Sa017", "Sa02",
                "Sa025", "Sa03", "Sa04", "Sa05", "Sa07", "Sa075", "Sa10",
                "Sa15", "Sa20", "Sa30", "Sa40", "Sa50", "Sa75", "Sa100"]
    columns = ["PGA", "PGV", "T0.010S", "T0.020S", "T0.030S", "T0.040S",
                "T0.050S", "T0.075S", "T0.100S", "T0.120S", "T0.150S",
                "T0.170S", "T0.200S", "T0.250S", "T0.300S", "T0.400S",
                "T0.500S", "T0.700S", "T0.750S", "T1.000S", "T1.500S",
                "T2.000S", "T3.000S", "T4.000S", "T5.000S", "T7.500S",
                "T10.000S",]
    periods = [0, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17,
            0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
            7.5, 10.0]
    
    # for i in range(len(targets)):
    #     _ = cut_period(folder, targets[i], periods[i], columns[i])

    # for index, m in enumerate(targets): # change columns name to target name (ex: T1.0S -> Sa10)
    #     TSMIP_df = pd.read_csv(f"../../../{folder}/TSMIP_FF_period/TSMIP_FF_{m}.csv")
    #     TSMIP_df = TSMIP_df.rename(columns={columns[index]: targets[index]})
    #     TSMIP_df.to_csv(f"../../../{folder}/TSMIP_FF_period/TSMIP_FF_{m}.csv", index=False)

    # for j in range(len(targets)):
    #     _ = pre_SMOGN(folder, targets[j])
    #     _ = synthesize(folder, targets[j])
    
    #? plot distributions
    # _ = SMOGN_plot("Sa01")
    
    #? statistics period
    total_files = glob.glob(r"../../../cut period_shallow crustal/TSMIP_FF_period/*.csv")
    all_df = []
    for file in total_files:
        all_df.append(pd.read_csv(file))
    _ = period_statistics(periods, all_df)
        
    
