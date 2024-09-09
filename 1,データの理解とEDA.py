import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#データフレーム定義
df = pd.read_csv("stock_price(Corrected).csv")
columns_to_scale = ["終値","始値","高値","安値","出来高(M)","変化率(%)"]

"""サンプル抽出
df_1990 = pd.read_csv("stock_price(1990).csv")
df_2000 = pd.read_csv("stock_price(2000).csv")
df_2010 = pd.read_csv("stock_price(2010).csv")
df_2020 = pd.read_csv("stock_price(2020).csv")

#10年ごと
filtering_1990 = df["日付け"].str.contains("1990")
filtering_2000 = df["日付け"].str.contains("2000")
filtering_2010 = df["日付け"].str.contains("2010")
filtering_2020 = df["日付け"].str.contains("2020")

#csv出力
df[filtering_1990].to_csv('stock_price(1990).csv',header=True,index=False)
df[filtering_2000].to_csv('stock_price(2000).csv',header=True,index=False)
df[filtering_2010].to_csv('stock_price(2010).csv',header=True,index=False)
df[filtering_2020].to_csv('stock_price(2020).csv',header=True,index=False)
"""

#基本統計量関数
def Basic_Statistics ():
    columns_A = ["終値","始値","高値","安値","出来高(M)","変化率(%)"]
    statistics = df.describe()
    print(statistics)
    print("var",end="    ")
    for column in columns_A:
        dr = df[column]
        print(format(dr.var(),"f"),end="  ")
    print("\nskew",end="      ")
    for column in columns_A:
        dr = df[column]
        print(format(dr.skew(),"f"),end="     ")
    print("\nkurt",end="      ")
    for column in columns_A:
        dr = df[column]
        print(format(dr.kurt(),"f"),end="     ")

#折れ線グラフ
def Line_Chart():
    fig,ax = plt.subplots()

    x  = df["日付け"]
    y1 = df["変化率(%)"]
    #y2 = df["出来高"]

    c1 = "red"
    #c2 = "blue"

    l1 = "Rate"
    #l2 = "Close"

    ax.set_xlabel("Date")
    ax.set_ylabel("Rate of change")
    ax.set_xticks(np.arange(0,len(df["日付け"]),365))
    ax.set_xticklabels(df["日付け"][np.arange(0,len(df["日付け"]),365)],rotation=45)
    ax.set_title("The rate of change over time")
    ax.grid()

    ax.plot(x,y1,color=c1,label=l1)
    #ax.plot(x,y2,color=c2,label=l2)

    ax.legend()
    fig.tight_layout()
    plt.show()

#ヒストグラム
def Histgram():
    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_subplot(111,xticks = range(30,320,20),xlabel="Price",ylabel="Days")

    data = df["終値"]
    n,bins,patches = ax.hist(data,bins=range(32,312,5))

    plt.show()

#散布図
def Scatter_Plot():
    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_subplot(111,xlabel="price",ylabel="Volume")

    ax.scatter(df["終値"],df["出来高(M)"])
    plt.show()

#散布図行列(外れ値検出用)
def Scatter_Plot_Matrix():
    sns.pairplot(df)
    plt.show()

#欠損値判定 
def Missing_Value():
    print(df.isnull().all())

#外れ値処理(使用済)
def Outlier():
    window_size = 60
    for column in df.columns[1:]:  # 最初の列は項目の文字列なので、スキップ
        df[f'{column}_MA'] = df[column].rolling(window=window_size).mean()
        df[f'{column}_STD'] = df[column].rolling(window=window_size).std()
        # 最初のウィンドウサイズ未満の部分を最初に利用可能な値で補完
        df[f'{column}_MA'].fillna(method='bfill', inplace=True)
        df[f'{column}_STD'].fillna(method='bfill', inplace=True)

        # 外れ値の条件設定
        df[f'{column}_Outlier'] = (df[column] > df[f'{column}_MA'] + 2 * df[f'{column}_STD']) | (df[column] < df[f'{column}_MA'] - 2 * df[f'{column}_STD'])

        # 外れ値を移動平均で置換
        df.loc[df[f'{column}_Outlier'], column] = round(df[f'{column}_MA'],1)

    df.drop(columns=[col for col in df.columns if '_MA' in col or '_STD' in col or '_Outlier' in col], inplace=True)
    df.to_csv('stock_price(Corrected_Outlier).csv', index=False)

Line_Chart()