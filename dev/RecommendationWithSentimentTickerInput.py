import yfinance as yf
from yahooquery import Ticker
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
def get_news_and_sentiment(symbol):
    ticker = Ticker(symbol)
    news = ticker.news
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    headlines = []
    if isinstance(news, list):
        for article in news[:5]:
            title = article.get('title', '')
            headlines.append(title)
            score = analyzer.polarity_scores(title)
            sentiment_scores.append(score['compound'])
    avg_sentiment = sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0
    return headlines, avg_sentiment

def get_community_sentiment(symbol):
    ticker = Ticker(symbol)
    try:
        trend = ticker.get_community_trend()
        sentiment = trend.get(symbol, {}).get('communitySentiment', {})
        bullish = sentiment.get('bullishPercent', 0)
        bearish = sentiment.get('bearishPercent', 0)
        return bullish - bearish
    except Exception:
        return 0

def make_recommendation(fundamentals, news_sentiment, community_sentiment):
    # Simple logic: combine PE, price position, and sentiment
    pe = fundamentals.get('PE Ratio')
    price = fundamentals.get('Current Price')
    low = fundamentals.get('52w Low')
    high = fundamentals.get('52w High')
    # If price is near 52w low, sentiment is positive, and PE is reasonable, suggest buy
    if price and low and high:
        price_pos = (price - low) / (high - low + 1e-6)
    else:
        price_pos = 0.5
    sentiment = 0.6 * news_sentiment + 0.4 * community_sentiment
    if pe and pe < 25 and price_pos < 0.4 and sentiment > 0.1:
        return "BUY (swing trade)"
    elif sentiment < -0.2 or (pe and pe > 40):
        return "SELL"
    else:
        return "HOLD"

if __name__ == "__main__":
    # List your stocks here
    symbols = ["UAMY", "PPTA", "ASTS", "TEM","RR","SOUND","LCID","SLV"]  # Example US stocks
    results = []
    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        fundamentals = get_fundamentals(symbol)
        headlines, news_sentiment = get_news_and_sentiment(symbol)
        community_sentiment = get_community_sentiment(symbol)
        recommendation = make_recommendation(fundamentals, news_sentiment, community_sentiment)
        print("Fundamentals:", fundamentals)
        print("Recent News Headlines:", headlines)
        print(f"News Sentiment: {news_sentiment:.2f}")
        print(f"Community Sentiment: {community_sentiment:.2f}")
        print("Recommendation:", recommendation)
        results.append({
            **fundamentals,
            "News Sentiment": news_sentiment,
            "Community Sentiment": community_sentiment,
            "Recommendation": recommendation
        })
        time.sleep(2)  # To avoid rate limits

    df = pd.DataFrame(results)
    print("\nSummary Table:")
    print(df[["Symbol", "Current Price", "PE Ratio", "52w High", "52w Low", "News Sentiment", "Community Sentiment", "Recommendation"]].to_string(index=False))