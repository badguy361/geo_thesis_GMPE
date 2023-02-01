import smogn
import pandas as pd

## load data
target = "Sa40"
TSMIP_df = pd.read_csv(f"../../../TSMIP_FF_{target}.csv")

## conduct smogn ##這步大概要兩個小時
TSMIP_smogn = smogn.smoter(
    data = TSMIP_df,
    y = 'T4.000S'
)
TSMIP_smogn.to_csv(f"../../../TSMIP_smogn_{target}.csv",index=False)
