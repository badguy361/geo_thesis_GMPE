import pandas as pd
# test_df=pd.read_csv("test_df.csv")
train_df=pd.read_csv("SMOGN_train.csv")
all_df=pd.read_csv("all_df.csv")

# test_aligned = test_df[train_df.columns]
# matched_rows = pd.merge(test_aligned, train_df, how='inner')

merged_df = pd.merge(train_df, all_df, how='right', indicator=True)
only_in_df2 = merged_df[merged_df['_merge'] == 'right_only']
only_in_df2 = only_in_df2.drop(columns=['_merge'])
random_rows = only_in_df2.sample(n=4369)
random_rows.to_csv("final_test.csv",index=False)
