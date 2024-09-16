import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
main_dir = "C:/Users/suram/OneDrive/Desktop/Code/Stock Performance Analyzer/"
prices = pd.read_csv(main_dir + "sp500_stocks.csv", parse_dates=['Date'], infer_datetime_format='%d/%m/%Y').sort_values(by=['Symbol', 'Date'], ignore_index=True)
stocks = pd.read_csv(main_dir + "sp500_companies.csv")

# Merge and prepare data
sp500 = pd.merge(prices, stocks, on='Symbol')
sp500['Return'] = sp500.groupby('Symbol')['Adj Close'].pct_change()
sp500['Volatility'] = sp500.groupby('Symbol')['Return'].rolling(window=30).std().reset_index(level=0, drop=True)

# RSI calculation function
def compute_rsi(data, period=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Apply RSI strategy
def apply_rsi_strategy(data, period=14, overbought=70, oversold=30):
    data['RSI'] = compute_rsi(data, period)
    data['Signal'] = 0
    data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy signal
    data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell signal
    data['Position'] = data['Signal'].shift(1)
    data['Strategy_Return'] = data['Return'] * data['Position']
    data['Cumulative_Return'] = (1 + data['Return']).cumprod() - 1
    data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod() - 1
    return data

# Sidebar for user input
st.sidebar.title('Stock Performance Analyzer')

# Get the list of symbols available in the prices dataset
available_symbols = sp500['Symbol'].unique()
company_names = stocks.set_index('Symbol')['Shortname']
available_names = company_names[company_names.index.isin(available_symbols)].to_dict()

selected_stock = st.sidebar.selectbox('Select a stock:', options=list(available_names.values()))
plot_option = st.sidebar.radio('Select graph to display:', options=['50-Day Moving Average', '200-Day Moving Average', 'Relative Strength Index'])

# Map the selected stock name back to the symbol
selected_symbol = [symbol for symbol, name in available_names.items() if name == selected_stock][0]

# Data preparation for selected stock
stock_data = sp500[sp500['Symbol'] == selected_symbol]

# Apply RSI strategy
stock_data_rsi = apply_rsi_strategy(stock_data.copy())

# Plotting function for RSI
def plot_rsi(data, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Adj Close'], label='Adjusted Close Price')
    plt.axhline(y=70, color='red', linestyle='--', label='RSI 70 (Overbought)')
    plt.axhline(y=30, color='green', linestyle='--', label='RSI 30 (Oversold)')
    plt.plot(data['Date'], data['RSI'], label='RSI', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price / RSI')
    plt.title(f'{symbol} RSI')
    plt.legend()
    st.pyplot(plt)

# Plotting function for Moving Averages
def plot_moving_averages(data, symbol, ma_period):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Adj Close'], label='Adjusted Close Price')
    plt.plot(data['Date'], data[f'MA_{ma_period}'], label=f'{ma_period}-Day Moving Average', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{symbol} {ma_period}-Day Moving Average')
    plt.legend()
    st.pyplot(plt)

# Calculate moving averages
stock_data['MA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
stock_data['MA_200'] = stock_data['Adj Close'].rolling(window=200).mean()

# Display selected graph
if plot_option == '50-Day Moving Average':
    plot_moving_averages(stock_data, selected_symbol, 50)
elif plot_option == '200-Day Moving Average':
    plot_moving_averages(stock_data, selected_symbol, 200)
elif plot_option == 'Relative Strength Index':
    plot_rsi(stock_data_rsi, selected_symbol)
