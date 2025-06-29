# 01_data_prep.py
import akshare as ak
import pandas as pd
import talib
from  datetime  import  datetime, timedelta

symbol =  "300350"
start_date = (datetime.today() - timedelta(days=365  *  2)).strftime('%Y%m%d')
df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date='20250520')

df = df.rename(columns={
       "日期":  "date",  "开盘":  "open",  "收盘":  "close",
       "最高":  "high",  "最低":  "low",  "成交量":  "volume"
})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 技术指标
df['rsi'] = talib.RSI(df['close'], timeperiod=14)
macd, macdsignal, macdhist = talib.MACD(df['close'],  12,  26,  9)
df['macd'], df['macds'], df['macdh'] = macd, macdsignal, macdhist

slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],  9,  3,  0,  3,  0)
df['kdjk'], df['kdjd'], df['kdjj'] = slowk, slowd,  3  * slowk -  2  * slowd

df = df.dropna().reset_index(drop=True)
df.to_csv("01data.csv", index=False)