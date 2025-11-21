import yahooquery as yq
import yfinance as yf
import pandas as pd
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import time
import requests
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


def write_to_snowflake(df, table_name, user, password, account, warehouse, database, schema, role):
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
        role=role
    )
    cs = conn.cursor()
    try:
        # Create table if not exists (simple schema, adjust as needed)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            SYMBOL STRING,
            CURRENT_PRICE FLOAT,
            TODAY_GAIN_LOSS FLOAT,
            NEWS_SENTIMENT FLOAT,
            SOCIAL_SENTIMENT FLOAT,
            SUGGESTED_BUY_DATE TIMESTAMP_NTZ,
            SUGGESTED_SELL_DATE TIMESTAMP_NTZ,
            NEXT_5D_FORECAST ARRAY
        )
        """
        cs.execute(create_table_sql)

        # Insert data
        # insert_sql = f"""
        # INSERT INTO {table_name} (Symbol, Current_Price, Today_Gain_Loss, News_Sentiment, Suggested_Buy_Date, Suggested_Sell_Date, Next_5d_Forecast)
        # VALUES (%s, %s, %s, %s, %s, %s, %s)
        # """
        # for _, row in df.iterrows():
        #     cs.execute(insert_sql, (
        #         row['Symbol'],
        #         row['Current Price'],
        #         row['Today % Gain/Loss'],
        #         row['News Sentiment'],
        #         row['Suggested Buy Date'],
        #         row['Suggested Sell Date'],
        #         row['Next 5d Forecast']
        #     ))

        success, nchunks, nrows, _ = write_pandas(conn, df, table_name, auto_create_table=False, overwrite=False)

        conn.commit()
    finally:
        cs.close()
        conn.close()

def get_active_stocks():
    screener = yq.Screener()
    results = screener.get_screeners(['most_actives','day_gainers'], count=1)
    stocks = results['most_actives']['quotes']
    return [stock['symbol'] for stock in stocks]

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    return ticker.history(period='1d')['Close'].iloc[-1]

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
        # Not enough data or no variation
        return pd.DataFrame({'ds': [], 'yhat': []})
    df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(5)

def get_news_sentiment(symbol):
    ticker = yq.Ticker(symbol)
    news = ticker.news
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    if isinstance(news, list):
        for article in news[:5]:
            score = analyzer.polarity_scores(article.get('title', ''))
            sentiments.append(score['compound'])
    return sum(sentiments)/len(sentiments) if sentiments else 0

def get_social_sentiment(symbol):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol.upper()}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; sentiment-scraper/1.0)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        messages = data.get("messages", [])

        bullish_count = 0
        bearish_count = 0

        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment", {})
            if sentiment:
                if sentiment.get("basic") == "Bullish":
                    bullish_count += 1
                elif sentiment.get("basic") == "Bearish":
                    bearish_count += 1

        total_sentiment = bullish_count + bearish_count
        if total_sentiment == 0:
            return 0  # No sentiment tags
        score = (bullish_count - bearish_count) / total_sentiment
        return round(score, 3)

    except Exception as e:
        print(f"Error fetching sentiment from Stocktwits for {symbol}: {e}")
        return 0


def suggest_entry_exit(forecast):
    min_row = forecast.loc[forecast['yhat'].idxmin()]
    max_row = forecast.loc[forecast['yhat'].idxmax()]
    date_format = "%Y-%m-%d"
    min_date = datetime.strptime(min_row['ds'], date_format).date()
    max_date = datetime.strptime(max_row['ds'], date_format).date()
    return min_date, max_date

if __name__ == "__main__":
    symbols = get_active_stocks()
    results = []
    for symbol in symbols:
        try:
            price = get_current_price(symbol)
            pct_gain_loss = get_today_gain_loss(symbol)
            forecast = predict_next_5_days(symbol)
            news_sent = get_news_sentiment(symbol)
            social_sent = get_social_sentiment(symbol)
            entry, exit = suggest_entry_exit(forecast)
            results.append({
                'SYMBOL': symbol,
                'CURRENT_PRICE': price,
                'TODAY_GAIN_LOSS': round(pct_gain_loss, 2) if pct_gain_loss is not None else None,
                'NEWS_SENTIMENT': round(news_sent, 2),
                'SOCIAL_SENTIMENT': round(social_sent, 2),
                'SUGGESTED_BUY_DATE': entry,
                'SUGGESTED_SELL_DATE': exit,
                'NEXT_5D_FORECAST': forecast['yhat'].tolist()
            })
            time.sleep(2)  # Add a 2-second delay between requests
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        df = pd.DataFrame(results)
    write_to_snowflake(
        df,
        'DAY_GAINERS_WITH_PREDICTION',
        'app_stock_data_gathering_usr',
        '2@UVwz8KvN!Qmsr',
        'XODOSPL-KRA09541',
        'SNOWFLAKE_LEARNING_WH',
        'APPDATA',
        'STOCK_MARKET_DATA',
        'APP_STOCK_DATA_GATHERING_USR_RW'
    )
    # df = pd.DataFrame(results)
    # print(df.to_string(index=False))