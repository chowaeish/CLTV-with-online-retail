import pandas as pd
import datetime as dt
import numpy as np
pip install lifetimes
pip install sqlalchemy
pip install mysql
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



df_ = pd.read_excel("online_retail_II.xlsx")
df = df_.copy()
df.head(5)
df=df[~df["Invoice"].str.contains("C",na=False)]
df=df[(df["Quantity"]>0)]
df.dropna(inplace=True)
df["Total_Price"]=df["Quantity"]*df["Price"]
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
today_date = dt.datetime(2010, 12, 11)
cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         "Invoice": lambda num: num.nunique(),
                                         "Total_Price": lambda Total_Price: Total_Price.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]
#BG-NBD Modelinin Kurulması
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])
#GAMMA-GAMMA Modelinin Kurulması
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)
#GÖrev1:
#BG-NBD ve GAMMA-GAMMA modeli ile CLTV'nin hesaplanması
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()
cltv.sort_values("clv", ascending=False).head(30)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values("clv", ascending=False).head(10)

scaler = MinMaxScaler(feature_range=(0, 1))
cltv_final["scaled_clv"] = scaler.fit_transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_clv", ascending=False).head()
#Alışveriş sıklığının en yüksek olması en kazançlı müşteri anlamına gelmiyormuş veya ortalama kazancın yüksek olması yine tek başına en iyi müşteri anlamına gelmiyor. Sonuç olarak hepsinin kombinasyonuyla en iyi sonuca varabiliriz.

#Görev 2:
#1ay:
bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

cltv_df.head()
#12ay:
bgf.predict(48,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)


cltv_df["expected_purc_12_months"] = bgf.predict(48,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

cltv_df.head()
#1ay ve 12 ay değerlerine baktığımızda ilk 10 müşteri sıramız değişmemiştir. Demek ki 1 aylık verimize bakarak 12 aylık şeklimizi tahmin edebiliriz.
#Görev3
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()


cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})
#SegmentD için: receny ve frequency ortalamaları düşük olduğu için kazançları en az segmenttir.
#SegmentA için: recency ve frequency ortamaları yüksek olduğu için doğal olarak daha fazla para harcamışlar ve bu yüzden clv leri en yüksek segment oluyorlar.
from sqlalchemy import create_engine
creds = {'user': 'synan_dsmlbc_group_8_admin',
         'passwd': 'iamthedatascientist*****!',
         'host': 'db.github.rocks',
         'port': 3306,
         'db': 'synan_dsmlbc_group_8'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
#
# # sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))
#

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)
cltv_final.to_sql(name='Hüseyin_Yıldırım', con=conn, if_exists='replace', index=False)



