import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
prices = pd.read_csv("https://raw.githubusercontent.com/ss6708/Stock-Analyzer-Graphing-Tool/main/sp500_stocks.csv", parse_dates=['Date'], infer_datetime_format='%d/%m/%Y').sort_values(by=['Symbol', 'Date'], ignore_index=True)
stocks = pd.read_csv("https://raw.githubusercontent.com/ss6708/Stock-Analyzer-Graphing-Tool/main/sp500_companies.csv")

# Merge and prepare data
sp500 = pd.merge(prices, stocks, on='Symbol')
sp500['Return'] = sp500.groupby('Symbol')['Adj Close'].pct_change()
sp500['Volatility'] = sp500.groupby('Symbol')['Return'].rolling(window=30).std().reset_index(level=0, drop=True)

# UI
st.title('Stock Performance Analyzer')
available_symbols = sp500['Symbol'].unique()
company_names = stocks.set_index('Symbol')['Shortname']
available_names = company_names[company_names.index.isin(available_symbols)].to_dict()

selected_stocks = st.sidebar.multiselect(
    'Select stocks to compare:', 
    options=list(available_names.values()), 
    default=list(available_names.values())[:5]  # Default to the first 5 stocks
)

# Map selected stock names back to their symbols
selected_symbols = [symbol for symbol, name in available_names.items() if name in selected_stocks]

# Prepare data for the selected stocks
filtered_data = sp500[sp500['Symbol'].isin(selected_symbols)]

# Calculate returns and cumulative returns
filtered_data['Return'] = filtered_data.groupby('Symbol')['Adj Close'].pct_change(fill_method=None)

# Calculate cumulative returns
filtered_data['Cumulative_Return'] = (
    filtered_data.groupby('Symbol')['Return']
    .apply(lambda x: (1 + x).cumprod() - 1)
    .reset_index(level=0, drop=True)  # Reset index to align with the original DataFrame
)

# Plot cumulative returns
def plot_cumulative_returns(data, symbols):
    plt.figure(figsize=(14, 7))
    for symbol in symbols:
        stock_data = data[data['Symbol'] == symbol]
        plt.plot(stock_data['Date'], stock_data['Cumulative_Return'], label=f"{symbol}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    st.pyplot(plt)

# Display the graph
if st.sidebar.checkbox('Show Cumulative Returns Comparison', value=True):
    plot_cumulative_returns(filtered_data, selected_symbols)
