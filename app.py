import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytz
from datetime import datetime, timedelta

# Set the title of the app
st.title('Indian Stock Market Intraday Options Trading App')

# Function to fetch intraday options data
def fetch_intraday_options_data(ticker, option_type='call', expiration='2025-03-15', strike_price=None):
    try:
        stock = yf.Ticker(ticker)
        # First check if options exist for this ticker
        available_expirations = stock.options
        if not available_expirations:
            st.error(f"No options available for {ticker}")
            return pd.DataFrame()  # Return empty DataFrame
            
        # If requested expiration isn't available, use the nearest one
        if expiration not in available_expirations:
            st.warning(f"Expiration {expiration} not available. Available dates: {available_expirations}")
            if available_expirations:
                expiration = available_expirations[0]  # Use the first available expiration
                st.info(f"Using {expiration} instead")
            else:
                return pd.DataFrame()
                
        options = stock.option_chain(expiration)
        if option_type == 'call':
            data = options.calls
        else:
            data = options.puts
        if strike_price and not data.empty:
            # Check if the exact strike price exists, if not use the nearest one
            if strike_price not in data['strike'].values:
                nearest_strike = data['strike'].iloc[(data['strike'] - strike_price).abs().argsort()[0]]
                st.warning(f"Strike price {strike_price} not available. Using nearest strike: {nearest_strike}")
                data = data[data['strike'] == nearest_strike]
            else:
                data = data[data['strike'] == strike_price]
        data['ticker'] = ticker
        data['option_type'] = option_type
        data['expiration'] = expiration
        return data
    except Exception as e:
        st.error(f"Error fetching options data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame

# Function to calculate moving averages for intraday options
def calculate_moving_averages(data, short_window=15, long_window=60):
    if data.empty:
        return data
    data['Short_MA'] = data['close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['close'].rolling(window=long_window).mean()
    return data

# Function to implement moving average crossover strategy for intraday options
def moving_average_crossover_strategy(data):
    if data.empty:
        return data
    data['Signal'] = 0
    data['Signal'][data['Short_MA'] > data['Long_MA']] = 1
    data['Signal'][data['Short_MA'] < data['Long_MA']] = -1
    data['Position_Change'] = data['Signal'].diff().fillna(0)
    data['Signal_Time'] = np.nan
    last_signal_time = None
    for i in range(len(data)):
        if data['Position_Change'].iloc[i] != 0:
            last_signal_time = data.index[i]
        if last_signal_time is not None:
            data['Signal_Time'].iloc[i] = last_signal_time
    return data

# Function to calculate returns for options
def calculate_returns(data):
    if data.empty:
        return data
    data['Return'] = np.log(data['close'] / data['close'].shift(1))
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)
    return data

# Function to implement stop-loss and take-profit for options
def implement_stop_loss_take_profit(data, stop_loss_price, take_profit_price):
    if data.empty:
        return data
    data['Stop_Loss'] = stop_loss_price
    data['Take_Profit'] = take_profit_price
    return data

# Function to calculate additional technical indicators for risk assessment
def calculate_additional_indicators(data):
    if data.empty:
        return data
    data['High-Low'] = data['high'] - data['low']
    data['High-Close'] = np.abs(data['high'] - data['close'].shift())
    data['Low-Close'] = np.abs(data['low'] - data['close'].shift())
    data['TR'] = np.max([data['High-Low'], data['High-Close'], data['Low-Close']], axis=0)
    data['ATR'] = data['TR'].rolling(window=14).mean()
    data['ATR_Percent'] = data['ATR'] / data['close'] * 100
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    plus_dm = np.where((data['high'].diff() > 0) & (data['high'].diff() > -data['low'].diff()), data['high'].diff(), 0)
    minus_dm = np.where((data['low'].diff() < 0) & (-data['low'].diff() > data['high'].diff()), -data['low'].diff(), 0)
    tr = data['TR']
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / data['ATR'])
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / data['ATR'])
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    data['ADX'] = pd.Series(dx).rolling(window=14).mean()
    data['Volume_MA'] = data['volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['volume'] / data['Volume_MA']
    return data

# Function to assess market conditions for options
def assess_market_conditions(data):
    if data.empty:
        return {}, 0
    latest = data.iloc[-1]
    atr_percent = latest['ATR_Percent']
    volatility = "Low" if atr_percent < 1 else "Medium" if atr_percent < 2 else "High"
    adx = latest['ADX']
    trend_strength = "Weak/Choppy" if adx < 20 else "Moderate" if adx < 40 else "Strong"
    market_direction = "Uptrend" if latest['Short_MA'] > latest['Long_MA'] else "Downtrend"
    volume_ratio = latest['Volume_Ratio']
    volume_strength = "Below Average" if volume_ratio < 0.8 else "Average" if volume_ratio < 1.2 else "Above Average"
    rsi = latest['RSI']
    momentum = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
    market_conditions = {
        "Volatility": volatility,
        "Trend Strength": trend_strength,
        "Market Direction": market_direction,
        "Volume": volume_strength,
        "Momentum": momentum
    }
    overall_score = (1 if volatility == "Low" else 2 if volatility == "Medium" else 3) + \
                    (1 if trend_strength == "Weak/Choppy" else 2 if trend_strength == "Moderate" else 3) + \
                    (3 if market_direction == "Uptrend" else 2) + \
                    (1 if volume_strength == "Below Average" else 2 if volume_strength == "Average" else 3) + \
                    (1 if momentum == "Oversold" or momentum == "Overbought" else 2 if momentum == "Neutral" else 3)
    trade_quality = (overall_score / 15) * 100
    return market_conditions, trade_quality

# Function to create pre-trade checklist for options
def create_pre_trade_checklist(data, ticker):
    if data.empty:
        st.warning("No data available to create pre-trade checklist")
        return
    market_conditions, trade_quality = assess_market_conditions(data)
    latest = data.iloc[-1]
    current_signal = latest['Signal']
    signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD"
    st.subheader("Pre-Trade Checklist")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Market Conditions")
        for key, value in market_conditions.items():
            st.markdown(f"**{key}:** {value}")
    with col2:
        st.markdown("### Trade Quality Assessment")
        trade_color = "green" if trade_quality >= 70 else "orange" if trade_quality >= 50 else "red"
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Overall Trade Quality</h4>
            <div style="margin: 0 auto; width: 200px; height: 200px; border-radius: 50%; background: conic-gradient({trade_color} {trade_quality}%, #f0f0f0 0); display: flex; align-items: center; justify-content: center;">
                <div style="background: white; width: 150px; height: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                    <h2 style="color: {trade_color};">{trade_quality:.1f}%</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    recommendation = "Strong BUY Signal" if signal_text == "BUY" and trade_quality >= 70 else \
                     "Moderate BUY Signal - Use Caution" if signal_text == "BUY" and trade_quality >= 50 else \
                     "Weak BUY Signal - Consider Skipping" if signal_text == "BUY" else \
                     "Strong SELL Signal" if signal_text == "SELL" and trade_quality >= 70 else \
                     "Moderate SELL Signal - Use Caution" if signal_text == "SELL" and trade_quality >= 50 else \
                     "Weak SELL Signal - Consider Skipping" if signal_text == "SELL" else \
                     "HOLD - No Trade Recommended"
    rec_color = "green" if trade_quality >= 70 else "orange" if trade_quality >= 50 else "red"
    st.markdown(f"<h3 style='text-align: center; color: {rec_color};'>{recommendation}</h3>", unsafe_allow_html=True)
    st.markdown("### Key Risk Factors")
    risk_factors = []
    if market_conditions["Volatility"] == "High":
        risk_factors.append("High market volatility - consider reducing position size")
    if market_conditions["Trend Strength"] == "Weak/Choppy":
        risk_factors.append("Market is choppy - expect false signals")
    if market_conditions["Volume"] == "Below Average":
        risk_factors.append("Low volume - potential for slippage")
    if market_conditions["Momentum"] == "Overbought" and signal_text == "BUY":
        risk_factors.append("Market is overbought - potential reversal risk")
    if market_conditions["Momentum"] == "Oversold" and signal_text == "SELL":
        risk_factors.append("Market is oversold - potential reversal risk")
    if risk_factors:
        for factor in risk_factors:
            st.markdown(f"- :warning: {factor}")
    else:
        st.markdown("- No significant risk factors identified")

# Function to create Signal Dashboard for options
def create_signal_dashboard(data, ticker, stop_loss_price, take_profit_price):
    if data.empty:
        st.warning("No data available to create signal dashboard")
        return
    latest_data = data.iloc[-1]
    current_price = latest_data['close']
    current_signal = latest_data['Signal']
    signal_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD"
    signal_color = "green" if current_signal == 1 else "red" if current_signal == -1 else "orange"
    signal_time = latest_data['Signal_Time']
    minutes_active = (data.index[-1] - signal_time).total_seconds() / 60 if pd.notnull(signal_time) else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<h1 style='text-align: center; color: {signal_color};'>{signal_text}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Current Price: ₹{current_price:.2f}</p>", unsafe_allow_html=True)
    with col2:
        if pd.notnull(signal_time):
            st.markdown("<h3 style='text-align: center;'>Signal Active Since</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{signal_time.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center;'>No Active Signal</h3>", unsafe_allow_html=True)
    with col3:
        st.markdown("<h3 style='text-align: center;'>Signal Duration</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{int(minutes_active)} minutes</p>", unsafe_allow_html=True)
    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("<h3 style='text-align: center;'>Profit Target</h3>", unsafe_allow_html=True)
        profit_target = take_profit_price
        distance_to_target = ((profit_target / current_price) - 1) * 100
        st.markdown(f"<p style='text-align: center;'>₹{profit_target:.2f} ({distance_to_target:.2f}% away)</p>", unsafe_allow_html=True)
    with col5:
        st.markdown("<h3 style='text-align: center;'>Stop Loss Level</h3>", unsafe_allow_html=True)
        stop_loss = stop_loss_price
        distance_to_stop = ((current_price / stop_loss) - 1) * 100
        st.markdown(f"<p style='text-align: center;'>₹{stop_loss:.2f} ({distance_to_stop:.2f}% away)</p>", unsafe_allow_html=True)

# Function to plot intraday options data with signal indicators
def plot_intraday_data(data, ticker):
    if data.empty:
        st.warning("No data available to plot")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['close'], label='Close Price')
    ax.plot(data.index, data['Short_MA'], label='Short MA')
    ax.plot(data.index, data['Long_MA'], label='Long MA')
    buy_signals = data[data['Position_Change'] == 1]
    ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    sell_signals = data[data['Position_Change'] == -1]
    ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    if len(buy_signals) > 0:
        last_buy = buy_signals.iloc[-1]
        ax.axhline(y=last_buy['Take_Profit'], color='green', linestyle='--', alpha=0.5)
        ax.text(data.index[-1], last_buy['Take_Profit'], 'Profit Target', verticalalignment='bottom', horizontalalignment='right', color='green')
        ax.axhline(y=last_buy['Stop_Loss'], color='red', linestyle='--', alpha=0.5)
        ax.text(data.index[-1], last_buy['Stop_Loss'], 'Stop Loss', verticalalignment='top', horizontalalignment='right', color='red')
    ax.set_title(f'{ticker} Intraday Options Price and Trading Signals')
    ax.set_xlabel('Time (IST)')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if data.index.size > 0:  # Check if there is data with valid index
        try:
            current_day = data.index[-1].date()
            day_data = data[data.index.date == current_day]
            if not day_data.empty:
                day_start = day_data.index[0]
                day_rect = patches.Rectangle((day_start, ax.get_ylim()[0]), data.index[-1] - day_start, ax.get_ylim()[1] - ax.get_ylim()[0], facecolor='yellow', alpha=0.1)
                ax.add_patch(day_rect)
        except (AttributeError, TypeError) as e:
            st.warning(f"Could not highlight current day: {str(e)}")
    st.pyplot(fig)

# Function to plot intraday strategy returns for options
def plot_intraday_strategy_returns(data):
    if data.empty:
        st.warning("No data available to plot strategy returns")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Strategy_Return'].cumsum(), label='Strategy Returns')
    ax.set_title('Cumulative Intraday Strategy Returns')
    ax.set_xlabel('Time (IST)')
    ax.set_ylabel('Returns')
    ax.legend()
    st.pyplot(fig)

# Function to screen stocks based on criteria
def screen_stocks(tickers, criteria):
    screened_stocks = []
    for ticker in tickers:
        try:
            data = fetch_intraday_options_data(ticker, strike_price=criteria['strike_price'])
            if not data.empty:
                data = calculate_moving_averages(data)
                last_data = data.iloc[-1]
                if (last_data['volume'] > criteria['min_volume'] and
                    last_data['Short_MA'] > last_data['Long_MA']):
                    screened_stocks.append(ticker)
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")
    return screened_stocks

# Main function to run the app
def main():
    # Add debug mode in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    ticker = st.text_input('Enter Stock Ticker (e.g., TCS.NS for Tata Consultancy Services)', 'TCS.NS')
    option_type = st.selectbox('Select Option Type', ['call', 'put'])
    
    # In debug mode, show available expirations
    if debug_mode and ticker:
        with st.expander("Debug Information"):
            try:
                stock = yf.Ticker(ticker)
                available_expirations = stock.options
                st.write("Available option expirations for", ticker, ":", available_expirations)
                
                # Show info about the stock
                info = stock.info
                if info:
                    st.write("Stock Information:")
                    st.json(info)
            except Exception as e:
                st.error(f"Error fetching debug info: {str(e)}")
    
    expiration = st.text_input('Enter Expiration Date (YYYY-MM-DD)', '2025-03-15')
    strike_price = st.number_input('Enter Strike Price', value=100.0, step=0.1)
    
    st.sidebar.header("Strategy Parameters")
    short_ma = st.sidebar.slider("Short MA Period", min_value=5, max_value=50, value=15)
    long_ma = st.sidebar.slider("Long MA Period", min_value=20, max_value=200, value=60)
    
    st.sidebar.header("Risk Management")
    stop_loss_price = st.sidebar.number_input("Stop Loss Price", value=100.0, step=0.1)
    take_profit_price = st.sidebar.number_input("Take Profit Price", value=120.0, step=0.1)
    
    st.sidebar.header("Screening Criteria")
    min_volume = st.sidebar.slider("Minimum Volume", min_value=10000, max_value=10000000, value=1000000, step=10000)
    
    tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    criteria = {
        'min_volume': min_volume,
        'strike_price': strike_price
    }
    
    # Add a button to execute the analysis
    run_analysis = st.button("Run Analysis")
    
    if run_analysis:
        with st.spinner("Fetching data and running analysis..."):
            # Screen stocks
            st.subheader("Screened Stocks")
            screened_stocks = screen_stocks(tickers, criteria)
            if screened_stocks:
                st.write("Stocks meeting the criteria:")
                st.write(screened_stocks)
            else:
                st.info("No stocks meet the specified criteria. Try adjusting your parameters.")
            
            # Fetch and process data
            data = fetch_intraday_options_data(ticker, option_type, expiration, strike_price)
            
            if data.empty:
                st.warning(f"Could not retrieve options data for {ticker} with expiration {expiration}. Please check the ticker symbol and expiration date.")
            else:
                # Apply all the analysis functions
                try:
                    data = calculate_moving_averages(data, short_window=short_ma, long_window=long_ma)
                    data = moving_average_crossover_strategy(data)
                    data = calculate_returns(data)
                    data = implement_stop_loss_take_profit(data, stop_loss_price, take_profit_price)
                    data = calculate_additional_indicators(data)
                    
                    # Show the analysis results
                    create_signal_dashboard(data, ticker, stop_loss_price, take_profit_price)
                    create_pre_trade_checklist(data, ticker)
                    plot_intraday_data(data, ticker)
                    
                    st.subheader("Recent Trading Signals")
                    signal_changes = data[data['Position_Change'] != 0].tail(10)
                    if not signal_changes.empty:
                        signal_table = pd.DataFrame({
                            'Time': signal_changes.index.strftime('%Y-%m-%d %H:%M'),
                            'Price': signal_changes['close'].round(2),
                            'Signal': signal_changes['Signal'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'}),
                            'Profit Target': signal_changes['Take_Profit'].round(2),
                            'Stop Loss': signal_changes['Stop_Loss'].round(2)
                        })
                        st.table(signal_table)
                    else:
                        st.write("No recent signal changes")
                    
                    st.subheader("Recent Price Activity")
                    display_cols = ['open', 'high', 'low', 'close', 'volume', 'Short_MA', 'Long_MA']
                    st.write(data[display_cols].tail().round(2))
                    
                    st.subheader("Strategy Performance")
                    plot_intraday_strategy_returns(data)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    if debug_mode:
                        st.exception(e)

if __name__ == '__main__':
    main()
