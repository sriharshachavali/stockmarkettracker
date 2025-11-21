import os
import pandas as pd
from datetime import datetime, timedelta
import polygon_io_basics as pio

def analyze_single(symbol: str, years: int = 2) -> pd.DataFrame:
    """
    Fetch 2 years of daily aggregates for one symbol, compute signals, and return a single-row DataFrame.
    """
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=365 * years)
    df_daily = pio.get_daily_aggregates(symbol, from_date.isoformat(), to_date.isoformat())
    print(df_daily)
    signal_info = pio.compute_signals(df_daily)
    print(signal_info)
    ref = pio.get_reference(symbol)
    row = {
        "symbol": symbol,
        "current_price": signal_info.get("current_price"),
        "today_pct": signal_info.get("today_pct"),
        "sma20": signal_info.get("sma20"),
        "sma50": signal_info.get("sma50"),
        "rsi": signal_info.get("rsi"),
        "signal": signal_info.get("signal"),
        "reason": signal_info.get("reason"),
        "suggested_buy_time": signal_info.get("suggested_buy_time"),
        "suggested_sell_time": signal_info.get("suggested_sell_time"),
        "name": ref.get("results", {}).get("name") if ref.get("results") else None,
        "market": ref.get("results", {}).get("market") if ref.get("results") else None
    }
    return pd.DataFrame([row])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze single ticker using polygon_io_basics helpers.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol to analyze (e.g. AAPL)")
    parser.add_argument("--years", type=int, default=2, help="Years of history to fetch")
    args = parser.parse_args()

    if not os.getenv("POLYGON_API_KEY"):
        raise SystemExit("Set POLYGON_API_KEY environment variable before running.")

    result_df = analyze_single(args.symbol, years=args.years)
    pd.set_option("display.max_columns", None)
    print(result_df.to_string(index=False))