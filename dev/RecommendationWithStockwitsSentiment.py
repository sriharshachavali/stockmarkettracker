import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time

def get_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        'Symbol': symbol,
        'Current Price': info.get('currentPrice'),
        'PE Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'Market Cap': info.get('marketCap'),
        '52w High': info.get('fiftyTwoWeekHigh'),
        '52w Low': info.get('fiftyTwoWeekLow'),
        'Dividend Yield': info.get('dividendYield'),
        'Beta': info.get('beta'),
        'Sector': info.get('sector'),
        'Industry': info.get('industry')
    }

def get_stocktwits_sentiment(symbol):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        r = requests.get(url)
        data = r.json()
        messages = data.get('messages', [])
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        headlines = []
        for msg in messages[:20]:  # Analyze up to 20 recent messages
            body = msg.get('body', '')
            headlines.append(body)
            score = analyzer.polarity_scores(body)
            sentiment_scores.append(score['compound'])
        avg_sentiment = sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0
        return headlines, avg_sentiment
    except Exception as e:
        print(f"Error fetching Stocktwits data for {symbol}: {e}")
        return [], 0

def make_recommendation(fundamentals, social_sentiment):
    pe = fundamentals.get('PE Ratio')
    price = fundamentals.get('Current Price')
    low = fundamentals.get('52w Low')
    high = fundamentals.get('52w High')
    if price and low and high:
        price_pos = (price - low) / (high - low + 1e-6)
    else:
        price_pos = 0.5
    sentiment = social_sentiment
    if pe and pe < 25 and price_pos < 0.4 and sentiment > 0.1:
        return "BUY (swing trade)"
    elif sentiment < -0.2 or (pe and pe > 40):
        return "SELL"
    else:
        return "HOLD"

if __name__ == "__main__":
    symbols = ["UAMY", "PPTA", "ASTS", "TEM", "RR", "SOUND", "LCID", "SLV"]  # Example US stocks
    results = []
    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        fundamentals = get_fundamentals(symbol)
        headlines, social_sentiment = get_stocktwits_sentiment(symbol)
        recommendation = make_recommendation(fundamentals, social_sentiment)
        print("Fundamentals:", fundamentals)
        print("Recent Stocktwits Messages:", headlines[:3])
        print(f"Stocktwits Sentiment: {social_sentiment:.2f}")
        print("Recommendation:", recommendation)
        results.append({
            **fundamentals,
            "Stocktwits Sentiment": social_sentiment,
            "Recommendation": recommendation
        })
        time.sleep(2)  # To avoid rate limits

    df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df[["Symbol", "Current Price", "PE Ratio", "52w High", "52w Low", "Stocktwits Sentiment", "Recommendation"]].to_string(index=False))