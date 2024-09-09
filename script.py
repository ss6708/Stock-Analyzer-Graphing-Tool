# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:13:09 2024

@author: suram
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

main_dir = "C:/Users/suram/OneDrive/Desktop/Code/Stock Performance Analyzer/"

prices = pd.read_csv(main_dir + "sp500_stocks.csv", parse_dates=['Date'],infer_datetime_format='%d/%m/%Y').sort_values(by=['Symbol','Date'], ignore_index=True)
stocks = pd.read_csv(main_dir + "sp500_companies.csv")
sp500 = pd.merge(prices, stocks, on='Symbol')

# Data Exploration

# getting daily returns
sp500['Return'] = sp500.groupby('Symbol')['Adj Close'].pct_change()

# getting volatility
sp500['Volatility'] = sp500.groupby('Symbol')['Return'].rolling(window=30).std().reset_index(level=0, drop=True)

# getting moving averages
sp500['MA_50'] = sp500.groupby('Symbol')['Adj Close'].rolling(window=50).mean().reset_index(level=0, drop=True)
sp500['MA_200'] = sp500.groupby('Symbol')['Adj Close'].rolling(window=200).mean().reset_index(level=0, drop=True)

# example
stock = 'GOOGL'
stock_data = sp500[sp500['Symbol'] == stock]
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Date'], stock_data['Adj Close'], label='Adjusted Close Price')
plt.plot(stock_data['Date'], stock_data['MA_50'], label='50-Day Moving Average', color='orange')
plt.plot(stock_data['Date'], stock_data['MA_200'], label='200-Day Moving Average', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{stock} Price and Moving Averages')
plt.legend()
plt.show()

# Moving-average based Trading Strategy 

# defining signals
sp500['Signal'] = 0
sp500.loc[sp500['MA_50'] > sp500['MA_200'], 'Signal'] = 1  # Buy signal
sp500.loc[sp500['MA_50'] < sp500['MA_200'], 'Signal'] = -1  # Sell signal

# backtesting on historical data
sp500['Position'] = sp500.groupby('Symbol')['Signal'].shift(1)
sp500['Strategy_Return'] = sp500['Return'] * sp500['Position']

# cumulative returns
sp500['Cumulative_Return'] = (1 + sp500['Return']).groupby(sp500['Symbol']).cumprod() - 1
sp500['Cumulative_Strategy_Return'] = (1 + sp500['Strategy_Return']).groupby(sp500['Symbol']).cumprod() - 1

# example performance
plt.figure(figsize=(14, 7))
plt.plot(sp500[sp500['Symbol'] == stock]['Date'], sp500[sp500['Symbol'] == stock]['Cumulative_Return'], label='Buy and Hold Return')
plt.plot(sp500[sp500['Symbol'] == stock]['Date'], sp500[sp500['Symbol'] == stock]['Cumulative_Strategy_Return'], label='Strategy Return', color='red')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title(f'{stock} Cumulative Returns vs. Strategy Returns')
plt.legend()
plt.show()

# filtering for top n stocks
num = 20 #chosen number of stocks for portfolio
last_returns = sp500.groupby('Symbol')['Cumulative_Strategy_Return'].last().reset_index()
topn = last_returns.sort_values(by='Cumulative_Strategy_Return', ascending=False, ignore_index = 'TRUE').head(num)

topn_symbols = topn['Symbol'].tolist()
sp500_top = sp500[sp500['Symbol'].isin(topn_symbols)].reset_index(drop=True)
returns_topn = sp500_top.pivot(index='Date', columns='Symbol', values='Return').dropna()

expected_returns_topn = returns_topn.mean()
cov_matrix_topn = returns_topn.cov()

# Finding Optimal Portfolio Weights

def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.sum(weights * returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_volatility  # Negative Sharpe ratio for minimization

initial_weights = num * [1. / num]  # initial equal weights

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights must sum to 1
bounds = tuple((0, 1) for _ in range(num))  # weights between 0 and 1

results_topn = minimize(portfolio_performance, initial_weights, args=(expected_returns_topn, cov_matrix_topn),
                         method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = results_topn.x

# Output a table of the top 20 stocks and their corresponding optimal weights
optimal_portfolio = pd.DataFrame({
    'Symbol': topn_symbols,
    'Optimal Weight': optimal_weights
})

print("Optimal Portfolio Weights for Top 20 Stocks:")
print(optimal_portfolio)


