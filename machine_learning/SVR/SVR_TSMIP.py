import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
df = pd.read_csv("../../../TSMIP_FF.csv")
# df.head(3)

# 篩選資料 - mask方法
# df_mask_Vs30=abs(df['Vs30']-300) < 1
# df_mask_Mw=df['MW'] > 3
# df_mask_Mw=df['MW'] < 4
# filtered_df = df[df_mask_Vs30]
# filtered_df.head(10)

# 篩選資料
# filtered_df = df[(abs(df.Vs30-300)<1) & (df['MW'] > 3) & (df['MW'] < 5)]

x = df.loc[:,['Vs30','MW','Rrup']]
y = df['PGA']

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, random_state=10, train_size=0.8)

svr_rbf = SVR(C=1e3, kernel='rbf', gamma='auto')
svr_rbf.fit(x_train,y_train)
svr_predict=svr_rbf.predict(x_test)
x_test_tmp_df = pd.DataFrame(x_test)
x_test_tmp_df.sort_values(by=[1],inplace=True) # inplace 會改變id
x_test = x_test_tmp_df.to_numpy()

plt.scatter(x_test[:,1], y_test, color= 'black', label= 'Data') #數據點
plt.plot(x_test[:,1], svr_predict, color= 'red', label= 'RBF model') #迴歸線
plt.xlabel('Mw')
plt.ylabel('svr_predict')
plt.title('Support Vector Regression')
plt.legend()
plt.show()