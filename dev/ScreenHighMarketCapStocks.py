import yahooquery as yq
import pandas as pd
from datetime import datetime
 
def get_top_penny_stocks():
    # Define the screener criteria
    screener = yq.Screener()
    # Get the screener results
    results = screener.get_screeners("most_actives", count=100)

    # Extract the relavant data
    stocks = results['most_actives']['quotes']
    df = pd.DataFrame(stocks)

    # Extract the relevant data
    df = pd.DataFrame(stocks)
    #Filter for penny stocks (price <=5) and sort by volume and change percentage
    df = df[(df['regularMarketPrice']>=1) & (df['regularMarketPrice'] <= 5) & (df['regularMarketVolume'] >=1000000)]
    # Filter and sort the data
    df = df[['symbol', 'shortName', 'regularMarketPrice', 'regularMarketVolume', 'regularMarketChangePercent', 'marketCap' , 'averageAnalystRating','dividendYield','regularMarketDayRange','fiftyTwoWeekRange']]
    df = df.sort_values(by=['regularMarketVolume', 'regularMarketChangePercent'], ascending=False)
    # Group by hour
    df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:00:00')
    grouped = df.groupby('timestamp').apply(lambda x: x.to_dict(orient='records')).to_dict()
    return grouped
 
if __name__ == "__main__":
    top_penny_stocks = get_top_penny_stocks()
    for hour, stocks in top_penny_stocks.items():
        print(f"Hour: {hour}")
        for stock in stocks:
            print(stock)
 