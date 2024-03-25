import pandas as pd


if __name__ == '__main__':
    dataset_type = "cut period_shallow crustal"
    target = "PGA"
    TSMIP_df = pd.read_csv(
        f"../../../{dataset_type}/TSMIP_FF_period/TSMIP_FF_{target}.csv")
    TSMIP_df.groupby(by=['EQ_ID']).sum()