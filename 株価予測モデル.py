import pandas as pd
import numpy as np
from keras._tf_keras.keras.layers import Dense,LSTM,Input
from keras._tf_keras.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#データフレーム定義
df = pd.read_csv("stock_price(Corrected_Outlier).csv")

# データの正規化
scaler = MinMaxScaler(feature_range=(0, 1))
d_close = scaler.fit_transform(df["終値"].values.reshape(-1, 1))
d_volume = scaler.fit_transform(df["出来高(M)"].values.reshape(-1, 1))
d_rate = scaler.fit_transform(df["変化率(%)"].values.reshape(-1, 1))

d_close,d_volume,d_rate = d_close.reshape(-1),d_volume.reshape(-1),d_rate.reshape(-1)

Return_datasets = []

for i in range(0,len(d_close)):
    Return_datasets.append([d_close[i],d_volume[i],d_rate[i]])


#訓練データとテストデータを入れるためのリスト
X = []
y = []

#何日前のデータを参照するのかを考慮する値
timesteps = 60

#timesteps日前の値を取り込む
for i in range(timesteps,len(Return_datasets)):
    X.append(Return_datasets[i-timesteps:i])
    y.append(Return_datasets[i])

#データセットを分割
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#全てndarrayに変換
X_train,X_test,y_train,y_test = np.array(X_train).reshape(6399,60,3),np.array(X_test).reshape(2743,60,3),np.array(y_train),np.array(y_test)

#モデル構築
model = Sequential()
model.add(Input(shape=(timesteps, 3)))
model.add(LSTM(units=75,return_sequences=False))
model.add(Dense(units=3,activation="sigmoid"))

#モデルの設定
model.compile(optimizer="nadam",
              loss='mean_absolute_error',
              metrics=["accuracy"])

#モデルの適応
model.fit(X_train,y_train,
          epochs=100,
          batch_size=32, 
          validation_data=(X_test,y_test)
          )

# 未来のデータポイントのための空の配列
start_point = 15 #過去のデータ読み取り開始地点
end_point = 1   #データ読み取り終了地点

future_points = 14 

# 最後のデータポイントから未来の予測を開始
new_input = Return_datasets[-start_point:-end_point]
new_input = np.array(new_input).reshape(1,start_point-end_point,3)

#予測したデータを格納する配列
predicted_stock_price = []
predicted_stock_price = np.array(predicted_stock_price)

#モデルを適応させて予測したデータを次のデータとして予想する
for _ in range(0,future_points):
    #モデル適応
    pred = model.predict(new_input,verbose=0)
    pred = np.array(pred).reshape(1,1,3)
    #予測データ配列に格納
    predicted_stock_price = np.append(predicted_stock_price,pred[0,0,0])
    #元データの古いデータを削除して、新しいデータに加える
    new_input = np.delete(new_input,0,1)
    new_input = np.append(new_input,pred,1)

#グラフでプロットする際にデータの年月日を取得する
last_row = df["日付け"].shape[0]-1
last_date = df["日付け"][last_row-end_point]
predicted_dates = pd.date_range(start=last_date, periods=future_points + 1, freq='B')[1:]

#グラフをプロット
plt.figure(figsize=(10, 6))
plt.plot(predicted_dates,predicted_stock_price, color='red', marker='o', linestyle='-', linewidth=2, markersize=5, label="Predicted Price")
plt.plot(predicted_dates,d_close[-start_point:-end_point],color='blue', marker='o', linestyle='-', linewidth=2, markersize=5, label="Actual Price")
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()