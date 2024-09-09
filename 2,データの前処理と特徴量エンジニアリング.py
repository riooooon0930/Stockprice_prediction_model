import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#データフレーム定義
df = pd.read_csv("stock_price(Corrected_Outlier).csv")
columns_to_scale = ["終値","始値","高値","安値","出来高(M)","変化率(%)"]

#標準化
def standardize():
    ss = StandardScaler()
    df[columns_to_scale] = ss.fit_transform(df[columns_to_scale])
    df.to_csv("stock_price(standardized).csv",index=False)

#正規化
def normalize():
    ms = MinMaxScaler()
    df[columns_to_scale] = ms.fit_transform(df[columns_to_scale])
    df.to_csv("stock_price(normalization).csv",index=False)

#変化率二値化
def binarization():
    df["変化率(%)"] = df["変化率(%)"].apply(lambda x: 0 if x <= 0 else 1)
    df.to_csv("stock_price(Return_binarization).csv",index = False)

