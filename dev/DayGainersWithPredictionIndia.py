import pandas as pd
import yfinance as yf
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import snowflake.connector

def get_active_indian_stocks():
    # Example: Nifty 50 stocks (you can expand this list or use NSE/BSE APIs for more)
    # Yahoo Finance uses '.NS' for NSE stocks
    nifty50_symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS"
    ]
    return nifty50_symbols

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1d')
    if hist.empty:
        return None
    return hist['Close'].iloc[-1]

def get_today_gain_loss(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='2d')
    if len(hist) < 2:
        return None
    prev_close = hist['Close'].iloc[-2]
    today_close = hist['Close'].iloc[-1]
    pct_change = ((today_close - prev_close) / prev_close) * 100
    return pct_change

def predict_next_5_days(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='6mo')
    if hist.empty or hist['Close'].nunique() < 2:
        return pd.DataFrame({'ds': [], 'yhat': []})
    df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(5)

def get_news_sentiment(symbol):
    # Yahoo Finance news for Indian stocks is limited via API.
    # You can use web scraping or third-party APIs for more coverage.
    # Here, we use yfinance's ticker.info['longBusinessSummary'] for sentiment.
    ticker = yf.Ticker(symbol)
    info = ticker.info
    summary = info.get('longBusinessSummary', '')
    analyzer = SentimentIntensityAnalyzer()
    if summary:
        score = analyzer.polarity_scores(summary)
        return score['compound']
    return 0

def suggest_entry_exit(forecast):
    if forecast.empty:
        return None, None
    min_row = forecast.loc[forecast['yhat'].idxmin()]
    max_row = forecast.loc[forecast['yhat'].idxmax()]
    return min_row['ds'], max_row['ds']

def write_to_snowflake(df, table_name, user, password, account, warehouse, database, schema):
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cs = conn.cursor()
    try:
        # Create table if not exists (simple schema, adjust as needed)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Symbol STRING,
            Current_Price FLOAT,
            Today_Gain_Loss FLOAT,
            News_Sentiment FLOAT,
            Suggested_Buy_Date TIMESTAMP_NTZ,
            Suggested_Sell_Date TIMESTAMP_NTZ,
            Next_5d_Forecast ARRAY
        )
        """
        cs.execute(create_table_sql)

        # Insert data
        insert_sql = f"""
        INSERT INTO {table_name} (Symbol, Current_Price, Today_Gain_Loss, News_Sentiment, Suggested_Buy_Date, Suggested_Sell_Date, Next_5d_Forecast)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        for _, row in df.iterrows():
            cs.execute(insert_sql, (
                row['Symbol'],
                row['Current Price'],
                row['Today % Gain/Loss'],
                row['News Sentiment'],
                row['Suggested Buy Date'],
                row['Suggested Sell Date'],
                row['Next 5d Forecast']
            ))
        conn.commit()
    finally:
        cs.close()
        conn.close()

# Replace these with your Snowflake credentials and desired table info
SNOWFLAKE_USER = 'your_user'
SNOWFLAKE_PASSWORD = 'your_password'
SNOWFLAKE_ACCOUNT = 'your_account'
SNOWFLAKE_WAREHOUSE = 'your_warehouse'
SNOWFLAKE_DATABASE = 'your_database'
SNOWFLAKE_SCHEMA = 'your_schema'
SNOWFLAKE_TABLE = 'INDIAN_STOCK_PREDICTIONS'

if __name__ == "__main__":
    symbols = get_active_indian_stocks()
    results = []
    for symbol in symbols:
        try:
            price = get_current_price(symbol)
            pct_gain_loss = get_today_gain_loss(symbol)
            forecast = predict_next_5_days(symbol)
            news_sent = get_news_sentiment(symbol)
            entry, exit = suggest_entry_exit(forecast)
            results.append({
                'Symbol': symbol,
                'Current Price': price,
                'Today % Gain/Loss': round(pct_gain_loss, 2) if pct_gain_loss is not None else None,
                'News Sentiment': round(news_sent, 2),
                'Suggested Buy Date': entry,
                'Suggested Sell Date': exit,
                'Next 5d Forecast': forecast['yhat'].tolist() if not forecast.empty else []
            })
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    df = pd.DataFrame(results)
    write_to_snowflake(
        df,
        SNOWFLAKE_TABLE,
        SNOWFLAKE_USER,
        SNOWFLAKE_PASSWORD,
        SNOWFLAKE_ACCOUNT,
        SNOWFLAKE_WAREHOUSE,
        SNOWFLAKE_DATABASE,
        SNOWFLAKE_SCHEMA
    )