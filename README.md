### Stock Performance Analyzer

**[Web App](https://stock-analyzer-graphing-tool-ss6708.streamlit.app)**

**Overview**:
The Stock Performance Analyzer is a Streamlit web application designed for visualizing stock performance. It allows users to select an S&P 500 company and view 3 performance metrics for that stock. 

[Data source](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks)

**Features**:
1. **Stock Selection**: Choose a stock from the S&P 500 list. The app only includes stocks with available sample data on daily pricing numbers from Kaggle.
2. **Graphing Options**:
   - **RSI (Relative Strength Index)**: Visualize the RSI to assess overbought or oversold conditions.
   - **50-Day Moving Average**: Display the 50-day moving average to identify medium-term trends.
   - **200-Day Moving Average**: Show the 200-day moving average to understand long-term trends.

**How It Works**:
1. **Data Loading**: The app loads historical stock data and company information from CSV files. A feature to auto-download the daily-updated csv from kaggle and load it to the web app is in the works. 
2. **RSI Calculation**: Computes the RSI to identify potential buy or sell signals based on momentum.
3. **Moving Averages**: Calculates and plots the 50-day and 200-day moving averages to track stock price trends.
4. **Interactive Visualizations**: Select the type of graph you want to view (RSI, 50-day MA, or 200-day MA) from the sidebar.
