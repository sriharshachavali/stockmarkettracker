import yahooquery as yq
import pandas as pd
from datetime import datetime
import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

def Heyy ():
    # Define the screener criteria
    screener = yq.Screener()
    # Get the screener results
    results = screener.get_screeners(['most_actives', 'day_gainers'], count=100)

    # Extract the relevant data
    stocks = results['most_actives']['quotes']
    df = pd.DataFrame(stocks)

    # Filter for penny stocks (price <= 5) and sort by volume and change percentage
    df = df[(df['regularMarketPrice'] >= 0.5) & 
            (df['regularMarketPrice'] <= 5)]
# (df['regularMarketVolume'] >= 1000000)]
    df = df.sort_values(by=['regularMarketChangePercent'], ascending=False)

    # Select relevant columns
    df = df[['symbol', 'shortName', 'regularMarketPrice', 'regularMarketVolume', 
             'regularMarketChangePercent', 'marketCap']]
    return df

def fetch_news_for_stock(symbol):
    # Use YahooQuery to fetch news for the stock
    ticker = yq.Ticker(symbol)
    news = ticker.news(25)  # Fetch news for the stock
    print(f"News for {symbol}: {news}")
    # Check if news is valid and iterable
    if isinstance(news, list):  # Ensure news is a list
        return [{"title": article.get('title', 'No Title'), "link": article.get('link', 'No Link')} for article in news]
    else:
        print(f"Unexpected data format for news: {news}")
        return []

def analyze_stock_with_openai(news_articles):
    # Combine news articles into a single prompt
    prompt = "Here are some news articles about the stock:\n\n"
    for article in news_articles:
        prompt += f"- {article['title']} ({article['link']})\n"

    prompt += "\nBased on this news, is this stock a good investment? Provide a detailed analysis."

    # Use OpenAI API to analyze the news
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # Step 1: Get good penny stocks
    good_penny_stocks = get_good_penny_stocks()
    print("Good Penny Stocks to Consider:")
    print(good_penny_stocks)

    # # Step 2: Fetch news and analyze each stock
    # for _, row in good_penny_stocks.iterrows():
    #     symbol = row['symbol']
    #     print(f"\nFetching news for {symbol}...")
    #     news_articles = fetch_news_for_stock(symbol)
    #     if news_articles:
    #         print("News Articles:")
    #         for article in news_articles:
    #             print(f"- {article['title']} ({article['link']})")

    #         # Step 3: Analyze the stock using OpenAI
    #         print("\nAnalyzing the stock with OpenAI...")
    #         analysis = analyze_stock_with_openai(news_articles)
    #         print(f"OpenAI Analysis for {symbol}:\n{analysis}")
    #     else:
    #         print(f"No news found for {symbol}.")