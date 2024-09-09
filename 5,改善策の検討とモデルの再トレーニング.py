import pandas as pd
import numpy as np
from keras._tf_keras.keras.layers import Dense,LSTM,Input,Dropout
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras import regularizers
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split

#データフレーム定義
df = pd.read_csv("stock_price(standardized_Outlier).csv")
df_2 = pd.read_csv("stock_price(Return_binarization).csv")

def Stock_LSTM():
    #全ての株価のラベルの値をとる
    Return_data = df[["終値","出来高(M)","変化率(%)"]]
    Return_datasets = Return_data.values
    #正負で二値分類した変化率の値をとる
    Return_data_ans = df_2["変化率(%)"]
    Return_datasets_ans = Return_data_ans.values

    #訓練データとテストデータを入れるためのリスト
    X = []
    y = []

    #何日前のデータを参照するのかを考慮する値
    timesteps = 60

    #timesteps日前の値を取り込む
    for i in range(timesteps,len(Return_datasets)):
        X.append(Return_datasets[i-timesteps:i])
        y.append(Return_datasets_ans[i])

    #データセットを分割
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    #全てndarrayに変換
    X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)

    #モデルの構築
    model = Sequential()
    model.add(Input(shape=(timesteps, 3)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50,return_sequences=False))
    model.add(Dense(units=25,activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=1,activation="sigmoid"))

    #モデルのコンパイル
    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])

    #モデルを実行
    model.fit(X_train,y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test,y_test),
            verbose=0)

    #予測値を出してその値を二値分類
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)

    #accuracyとf値で評価
    accuracy = accuracy_score(y_test,predicted_classes)
    F_score  = f1_score(y_test,predicted_classes)

    print(f"Accuracy:{accuracy:.4f}")
    print(f"F-score:{F_score:.4f}")

    #結果
    #Accuracy:0.5468
    #F-score:0.4819

Stock_LSTM()