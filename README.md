# Trainee-Assessment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()

datafile = 'ETH-USD.csv'
data = pd.read_csv(datafile, index_col = 'Date')
data.index = pd.to_datetime(data.index)
data

df = data.copy()
sma_span = 50
ema_span = 20
df['sma50'] = df['Adj Close'].rolling(sma_span).mean()
df['ema20'] = df['Adj Close'].ewm(span=ema_span).mean()
df.round(3)

df.dropna(inplace = True)
df.round(3)

def plot_system1(data):
    df = data.copy()
    dates = df.index
    price = df['Adj Close']
    sma50 = df['sma50']
    ema20 = df['ema20']
    
    with plt.style.context('fivethirtyeight'):
        fig = plt.figure(figsize = (14,7))
        plt.plot(dates, price, linewidth = 1.5, label = 'CPB price - Daily Adj Close')
        plt.plot(dates, sma50, linewidth = 2, label = '50 SMA')
        plt.plot(dates, ema20, linewidth = 2, label = '20 EMA')
        plt.title ("A Simple Crossover System")
        plt.ylabel('Price($)')
        plt.legend()
        
    plt.show()
    
    plot_system1(df)
    
long_positions = np.where(df['ema20'] > df['sma50'], 1, 0)
df['Position'] = long_positions

df.round(3)

buy_signals = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
df.loc[buy_signals].round(3)

buy_signals_prev = (df['Position'].shift(-1) == 1) & (df['Position'] == 0)
df.loc[buy_signals | buy_signals_prev].round(3)

def plot_system1_sig(data):
    df = data.copy()
    dates = df.index
    price = df['Adj Close']
    sma50 = df['sma50']
    ema20 = df['ema20']
    
    buy_signals = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
    buy_marker = sma50 * buy_signals - (sma50.max()*.05)
    buy_marker = buy_marker[buy_signals]
    buy_dates = df.index[buy_signals]
    sell_signals = (df['Position'] == 0) & (df['Position'].shift(1) == 1)
    sell_marker = sma50 * sell_signals + (sma50.max()*.05)
    sell_marker = sell_marker[sell_signals]
    sell_dates = df.index[sell_signals]
    
    with plt.style.context('fivethirtyeight'):
        fig = plt.figure(figsize=(14,7))
        plt.plot(dates, price, linewidth=1.5, label='CPB price - Daily Adj Close')
        plt.plot(dates, sma50, linewidth=2, label='50 SMA')
        plt.plot(dates, ema20, linewidth=2, label='20 EMA')
        plt.scatter(buy_dates, buy_marker, marker='^', color='green', s=160, label='Buy')
        plt.scatter(sell_dates, sell_marker, marker='v', color='red', s=160, label='Sell')
        plt.title("A Simple Crossover System with Signals")
        plt.ylabel('Price($)')
        plt.legend()
    
    plt.show()
    
    plot_system1_sig(df)
    
df['Hold'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df['Strategy'] = df['Position'].shift(1) * df['Hold']
df.dropna(inplace=True)
df

returns = np.exp(df[['Hold', 'Strategy']].sum()) - 1
print(f"Buy and hold return: {round(returns['Hold']*100,2)}%")
print(f"Strategy return: {round(returns['Strategy']*100,2)}%")

n_days = len(df)
ann_returns = 252 / n_days * returns
print(f"Buy and hold annualized return: {round(ann_returns['Hold']*100,2)}%")
print(f"Strategy annualized return:{round(ann_returns['Strategy']*100,2)}%")

df.to_csv('Latest Result Cryptocurrency.csv', index = False)
