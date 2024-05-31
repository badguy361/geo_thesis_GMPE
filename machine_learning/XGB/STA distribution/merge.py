import pandas as pd
# sta = pd.read_csv("TSMIP_FF.csv")
# sta_info = pd.read_csv("station_id_info.csv")
# sta_info.head()

# merged_df=pd.merge(sta_info,sta,on='STA_ID',how="left")
# deduped_df = merged_df.drop_duplicates(subset=['STA_ID'], keep='first')
# print(deduped_df)
# deduped_df.to_csv("new.csv",index=False)

sta_info = pd.read_csv("station_id_info.csv")
station_id = 0
sta_info.iloc[station_id+1]["Vs30"]

