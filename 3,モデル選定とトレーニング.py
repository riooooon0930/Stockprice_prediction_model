#基本モジュール
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
#keras
from keras._tf_keras.keras.layers import Dense,LSTM,Input,Dropout,Conv2D,MaxPooling2D,Flatten
from keras._tf_keras.keras.models import Model,Sequential
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import regularizers

#sklearn
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#データフレーム定義
df = pd.read_csv("stock_price(standardized).csv")
df_2 = pd.read_csv("stock_price(Return_binarization).csv")

#終値を用いたLSTMモデル
def ClosePrice_LSTM():
    #終値の値をとる
    CPrice_data = df["終値"]
    CPrice_datasets = CPrice_data.values

    #訓練データとテストデータにそれぞれ分割
    train_data_len = int(np.ceil(len(CPrice_datasets)*0.7))
    train_data = CPrice_datasets[0:train_data_len]
    test_data = CPrice_datasets[train_data_len-60:]

    #訓練データとテストデータの値とラベル定義
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i]) #60単位でデータを区切る
        y_train.append(train_data[i])

    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i])
    y_test = CPrice_datasets[train_data_len:]

    #訓練データとテストデータをndarrayにキャストしLSTMに読み込ませる
    x_train,y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    #LSTMによる学習モデルの作成(Sequential)
    model = Sequential()
    model.add(LSTM(128,return_sequences = True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')

    """
    #LSTMによる学習モデルの作成(functionAPI)
    inputs = Input(shape=(x_train.shape[1],))
    x = LSTM(128)(inputs)
    x = LSTM(64)(x)
    x = Dense(25)(x)
    predictions = Dense(1)(x)
    model = Model(inputs = input,outputs = predictions)

    #モデルの処理を設定
    model.compile(loss="mean_squared_error",
                optimizer="adam",
                metrics=["accuracy"])
    """

    #モデルの実行
    model.fit(x=x_train,y=y_train,batch_size=1,epochs=1,verbose=1)

    #モデルの評価
    score = model.evaluate(x_test,y_test,verbose=0)
    print("Test loss:",score)

#直近60日の変化率の平均と正解ラベルの比較(他モデル比較用)
def Rand_1():
    #変化率の値をとる
    Return_data = df["変化率(%)"]
    Return_datasets = Return_data.values
    Return_data_ans = df_2["変化率(%)"]
    Return_datasets_ans = Return_data_ans.values

    #変化率を格納する配列
    x_data = []         #データ加工前の状態を格納
    x_classed_data = [] #データ加工後の二値分類された状態を格納
    y_data = []         #正解ラベルの二値分類された状態をを格納

    #変化率のデータをそれぞれ格納(60単位(2ヶ月)でデータを区切る)
    for i in range(60,len(Return_datasets)):
        x_data.append(Return_datasets[i-60:i]) #加工前の変化率
        y_data.append(Return_datasets_ans[i])  #正解ラベル

    #直近60日(2ヶ月)分の変化率の平均をとり、その値が正ならば1、負ならば0の値に分類し格納する。
    for i in range(0,len(y_data)):
        avg_return = x_data[i].mean()
        x_classed_data.append(1 if avg_return >=0 else 0)

    #データを加工して二値分類したデータと正解ラベルを比較し、accuracyとF-scoreを求める
    accuracy = accuracy_score(y_data,x_classed_data)
    f_score  = f1_score(y_data,x_classed_data)

    print(f"Accuracy:{accuracy:.4f}")
    print(f"F-score:{f_score:.4f}")
    #結果
    #Accuracy:0.5010
    #F-score:0.4644

#変化率の上昇(1)または下降(0)をランダムな値を出力しその値と正解ラベルの比較(他モデル比較用)
def Rand_2():
    #変化率を格納する配列
    x_data = [] #ランダムな値を格納
    y_data = [] #正解ラベルを格納

    #ランダムな値(0,1)と正解ラベルを格納
    for i in range(0,len(df_2["変化率(%)"].values)):
        x_data.append(random.choice([0,1]))
        y_data.append(df_2["変化率(%)"].values[i])

    #ランダムなデータと正解ラベルを比較し、accuracyとF-scoreを求める
    accuracy = accuracy_score(y_data,x_data)
    f_score  = f1_score(y_data,x_data)

    #print(f"Accuracy:{accuracy:.4f}")
    #print(f"F-score:{f_score:.4f}")
    return [accuracy,f_score]

#Rand_2を1000回行いその結果の平均をとる
def Rand_2_avg():
    acc = []
    f_score =[]

    for i in range(0,1000):
        result = Rand_2()
        acc.append(result[0])
        f_score.append(result[1])

    print(f"Accuracy:{(sum(acc)/len(acc)):.4f}")
    print(f"F-score:{(sum(f_score)/len(f_score)):.4f}")
    #結果
    #Accuracy:0.5000
    #F-score:0.4699

#ロジスティック回帰を用いた株価変動予測
def Logistic_Regression():
    #変化率の値をとる
    Return_data = df["変化率(%)"]
    Return_datasets = Return_data.values    #変化率の値(実際の変動値)
    Return_data_ans = df_2["変化率(%)"]
    Return_datasets_ans = Return_data_ans.values    #変化率の増減(増減を二値分類した値)

    #訓練データとテストデータにそれぞれ分割
    train_xdata_len = int(np.ceil(len(Return_datasets)*0.7))    #全体データの7割を訓練データに変換

    train_xdata = Return_datasets[0:train_xdata_len]    #訓練データ(特徴量)のデータセット
    test_xdata = Return_datasets[train_xdata_len-60:]   #テストデータ(特徴量)のデータセット

    train_ydata = Return_datasets_ans[0:train_xdata_len]    #訓練データ(ラベル)のデータセット
    test_ydata = Return_datasets_ans[train_xdata_len-60:]   #テストデータ(ラベル)のデータセット

    #特徴量とラベルの配列データ
    x_train = []
    x_test  = []
    y_train = []
    y_test  = []

    #訓練データのデータセットを分割(60単位(2ヶ月)でデータを区切る)
    for i in range(60,train_xdata_len):
        x_train.append(train_xdata[i-60:i]) #訓練データ(特徴量)
        y_train.append(train_ydata[i])      #訓練データ(ラベル)

    for i in range(60,len(test_xdata)):
        x_test.append(test_xdata[i-60:i])   #テストデータ(特徴量)
        y_test.append(test_ydata[i])        #テストデータ(ラベル)

    model = LogisticRegression()            #ロジスティック回帰のモデル構築
    model.fit(x_train,y_train)              #モデルの学習
    y_pred = model.predict(x_test)          #学習したモデルを使って予測

    acc = accuracy_score(y_test,y_pred)     #accuracyの算出
    f_score = f1_score(y_test,y_pred)       #F値の算出

    print(f"Accuracy{acc:.4f}")
    print(f"F-score:{f_score:.4f}")
    #結果
    #Accuracy0.5120
    #F-score:0.0274

#ランダムフォレストを用いた株価変動予測
def RandomForest():
    #特徴量に全てのデータの代入
    X = df[["終値","出来高(M)","変化率(%)"]]

    # 予測するラベルを作成
    # 今日の終値が前日の終値より高ければ1、低ければ0とする
    df['Target'] = (df['終値'].shift(-1) > df['終値']).astype(int)
    y = df['Target'].dropna()

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #モデルの構築と実行
    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train,y_train)
    #予測値の算出
    y_pred = model.predict(X_test)
    #予測値と正解ラベルを比較
    acc = accuracy_score(y_test,y_pred)
    f_score = f1_score(y_test,y_pred)

    #print(f"Accuracy:{acc:.4f}")
    #print(f"F-score:{f_score:.4f}")
    return [acc,f_score]

#RandomForestを100回行いその平均をとる
def RandomForest_avg():
    #AccuracyとF値を代入する配列作成
    acc = []
    f_score =[]

    for i in range(0,100):
        result = RandomForest()
        acc.append(result[0])
        f_score.append(result[1])

    print(f"Accuracy:{(sum(acc)/len(acc)):.4f}")
    print(f"F-score:{(sum(f_score)/len(f_score)):.4f}")
    #結果
    #Accuracy:0.5271
    #F-score:0.4162

#MLPを用いた株価変動予測
def Multilayer_Perceptron():
    #全ての株価のラベルの値をとる
    Return_data = df_2[["終値","出来高(M)","変化率(%)"]]
    Return_datasets = Return_data.values
    #二値化した変化率の値をとる
    Return_data_ans = df_2["変化率(%)"]
    Return_datasets_ans = Return_data_ans.values

    #加工後の特徴量とラベルデータを入れるそれぞれ入れる配列
    X = []  #加工後の特徴量データを入れる配列
    y = []  #加工後のラベルデータを入れる配列

    #変化率のデータをそれぞれ格納(60単位(2ヶ月)でデータを区切る)
    for i in range(60,len(Return_datasets)):
        X.append(Return_datasets[i-60:i]) #特徴量("終値","出来高(M)","変化率(%)")
        y.append(Return_datasets_ans[i])  #正解ラベル("変化率")

    #それぞれの特徴量とラベルデータを訓練データとテストデータに分割
    X_train,X_test,y_train,y_true = train_test_split(X,y,test_size=0.3)

    #分割したデータをMLPモデルに組み込むことができるようにデータの形を整理
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1,180)
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1,180)
    y_train = to_categorical(y_train,num_classes=2)
    y_test = to_categorical(y_true,num_classes=2)

    #データの形確認用コード
    #print("shape of X_train",np.shape(X_train))
    #print("shape of y_train",np.shape(y_train))

    #モデルの構築(Sequential型)
    model = Sequential()
    #4層のモデルでありReluを用いる
    model.add(Input(shape=(180,)))
    model.add(Dense(units=64,activation="relu"))
    model.add(Dense(units=64,activation="relu"))
    model.add(Dense(units=32,activation="relu"))
    model.add(Dense(units=2,activation="softmax"))
    #モデルの損失関数、最適化関数、評価関数の決定
    model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
    #モデルを実際の数値に当てはめる
    model.fit(X_train,y_train,
            batch_size=32,epochs=100,
            validation_data=(X_test,y_test))
    #モデルの内容を評価する
    test_loss, test_acc =  model.evaluate(X_test, y_test, verbose=2)

    #F値を算出するための準備
    #完成したモデルにX_testを代入しその値をy_predとして取り出す
    y_pred = model.predict(X_test)
    #取り出されたy_predの値をy_true(y_test)の値と揃える
    y_pred_classes = np.argmax(y_pred,axis=1)

    #F値の算出
    f_score = f1_score(y_true,y_pred_classes)

    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test F1 score: {f_score:.4f}')

    return[test_acc,f_score]

#MLPを100回行いその平均をとる
def Multilayer_Perceptron_avg():
    #AccuracyとF値を代入する配列作成
    acc = []
    f_score =[]

    for i in range(0,100):
        result = Multilayer_Perceptron()
        acc.append(result[0])
        f_score.append(result[1])

    #print(f"Accuracy:{(sum(acc)/len(acc)):.4f}")
    #print(f"F-score:{(sum(f_score)/len(f_score)):.4f}")
    #結果
    #Accuracy:0.5162
    #F-score:0.4431

#LSTMを用いた株価変動予測
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
    model.add(LSTM(units=64,return_sequences=True))
    model.add(LSTM(units=64,return_sequences=False))
    model.add(Dense(units=32,activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=1,activation="sigmoid"))

    #モデルのコンパイル
    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])

    #モデルを実行
    model.fit(X_train,y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test,y_test))

    #予測値を出してその値を二値分類
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)

    #accuracyとf値で評価
    accuracy = accuracy_score(y_test,predicted_classes)
    F_score  = f1_score(y_test,predicted_classes)

    print(f"Accuracy:{accuracy:.4f}")
    print(f"F-score:{F_score:.4f}")

    #結果
    #Accuracy:0.5275
    #F-score:0.4476

#ARIMAを用いた株価変動予測
def Stock_ARIMA():
    #変化率の値を取り出す
    C_Rate = df["変化率(%)"]
    #auto_arimaを用いて適切な(p,d,q)の値を算出(AICで最も良い記録を残した(1,0,2)を使用)
    stepwise_model = auto_arima(C_Rate, 
                                start_p=1, start_q=1,  # p, qの開始点
                                max_p=3, max_q=3,  # p, qの最大値
                                start_P=0, seasonal=False,
                                d=None, D=0,
                                max_d=5,  # dの最大値
                                trace=False,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)

    #トレーニングデータとテストデータに分割(今回はテストデータのみを用いる)
    train_size = int(len(df_2) * 0.8)
    test =  df_2["変化率(%)"][train_size:]

    #ARIMAモデルによる予測値を格納する配列
    prediction = []

    #データをtrainデータ全体を少しずつずらしてテストデータの要素を最初から1つずつ予測
    #予測したデータが正か負かを判定し、次の株価が上がるか下がるかを計算する
    #予測したデータをpredictionに追加する
    for i in range(0,len(test)):
        train = df["変化率(%)"][i:train_size+i]#予測するデータを少しずつすらす

        #モデルの適合
        model = ARIMA(train,order=stepwise_model.order)#(p,d,q)は先ほど算出した値を使用
        model_fit = model.fit()

        #次の予測値を算出し配列に格納していく
        forecast = model_fit.forecast(steps=1)
        prediction.append(1 if forecast.iloc[0] > 0 else 0)

    #accuracyとF値の値を比較
    accuracy = accuracy_score(test,prediction)
    F_score  = f1_score(test,prediction)

    #accuracyとF値の値を表示
    print(f"Accuracy:{accuracy:.4f}")
    print(f"F-score:{F_score:.4f}")
    #結果
    #Accuracy:0.5117
    #F-score:0.4889

os.system('afplay /System/Library/Sounds/Glass.aiff')
def beep(freq, dur=100):
    import os
    os.system('afplay /System/Library/Sounds/Glass.aiff')

beep(2000, 500)
