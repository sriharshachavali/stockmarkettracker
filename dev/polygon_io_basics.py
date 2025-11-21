import os
import time
import math
import requests
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from typing import List, Dict, Optional
import numpy as np

def _print_df(name: str, df: pd.DataFrame):
    """Helper for printing DataFrame summaries to stdout for debugging."""
    try:
        print(f"[polygon_io_basics] DataFrame '{name}' shape={getattr(df, 'shape', None)}")
        if df is None:
            print(f"[polygon_io_basics] DataFrame '{name}' is None")
            return
        if isinstance(df, pd.DataFrame) and df.empty:
            print(f"[polygon_io_basics] DataFrame '{name}' is empty")
            return
        # Print head (up to 20 rows) to avoid flooding
        print(df.head(20).to_string())
    except Exception as e:
        print(f"[polygon_io_basics] Failed to print DataFrame '{name}': {e}")

# --- Configuration ---
POLYGON_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_KEY:
    print("Warning: POLYGON_API_KEY not set. _get will raise unless set before calling endpoints.")
BASE = "https://api.polygon.io"
RATE_LIMIT_PER_MIN = 5  # polygon free/paid account typical policy in prompt
SLEEP_PER_CALL = 60.0 / RATE_LIMIT_PER_MIN + 0.2  # conservative spacing

# --- Helpers: requests with backoff ---
def _get(url: str, params: Dict = None, max_retries: int = 5) -> Optional[Dict]:
    headers = {}
    params = params or {}
    if not POLYGON_KEY:
        raise RuntimeError("POLYGON_API_KEY environment variable not set")
    params["apiKey"] = POLYGON_KEY
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    print(f"[polygon_io_basics] Invalid JSON response for {url}: {r.text[:500]}")
                    return None
            if r.status_code == 429:
                print(f"[polygon_io_basics] Rate limited (429) on attempt {attempt+1} for {url}. Sleeping {backoff}s.")
                time.sleep(backoff)
                backoff *= 2
                continue
            # show helpful debug info for other non-200 responses
            print(f"[polygon_io_basics] Unexpected status {r.status_code} for {url}: {r.text[:1000]}")
            return None
        except requests.RequestException as e:
            print(f"[polygon_io_basics] RequestException for {url}: {e}. retrying in {backoff}s")
            time.sleep(backoff)
            backoff *= 2
    print(f"[polygon_io_basics] Exceeded retries for {url}")
    return None

# --- Polygon data pullers ---
def get_all_tickers(limit: int = 1000) -> List[str]:
    """
    Fetch tickers via /v3/reference/tickers (paginated). Returns list of symbols.
    Keep the number modest because of rate limits.
    """
    symbols = []
    url = f"{BASE}/v3/reference/tickers"
    params = {"market": "stocks", "active": "true", "limit": 100}
    cursor = None
    while len(symbols) < limit:
        if cursor:
            params["cursor"] = cursor
        payload = _get(url, params=params)
        if not payload or "results" not in payload:
            break
        for r in payload["results"]:
            symbols.append(r["ticker"])
            if len(symbols) >= limit:
                break
        cursor = payload.get("next_url")
        if not cursor:
            break
        time.sleep(SLEEP_PER_CALL)
    return symbols[:limit]

def get_daily_aggregates(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Uses /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}
    Returns dataframe with columns: timestamp, open, high, low, close, volume
    """
    url = f"{BASE}/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    payload = _get(url, params=params)
    time.sleep(SLEEP_PER_CALL)
    if payload is None:
        print(f"[polygon_io_basics] No payload returned for {symbol} {from_date}->{to_date}. Check API key, rate limits, symbol.")
        return pd.DataFrame()
    if not isinstance(payload, dict):
        print(f"[polygon_io_basics] Unexpected payload type for {symbol}: {type(payload)}")
        return pd.DataFrame()
    if "results" not in payload:
        # polygon often returns {'status':'ERROR','error':'...'} or {'results':[]}
        print(f"[polygon_io_basics] 'results' missing for {symbol}. payload: {payload}")
        return pd.DataFrame()
    if not payload["results"]:
        print(f"[polygon_io_basics] 'results' empty for {symbol}. No data for range {from_date} to {to_date}.")
        df_empty = pd.DataFrame()
        _print_df(f"daily_aggregates_{symbol}_{from_date}_{to_date}", df_empty)
        return df_empty
    df = pd.DataFrame(payload["results"])
    if df.empty:
        _print_df(f"daily_aggregates_{symbol}_{from_date}_{to_date}", df)
        return df
    df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize(None)
    df.set_index("timestamp", inplace=True)
    _print_df(f"daily_aggregates_{symbol}_{from_date}_{to_date}", df)
    return df

def get_minute_aggregates(symbol: str, from_dt: str, to_dt: str) -> pd.DataFrame:
    """
    /v2/aggs/ticker/{symbol}/range/1/minute/{from}/{to}
    """
    url = f"{BASE}/v2/aggs/ticker/{symbol}/range/1/minute/{from_dt}/{to_dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    payload = _get(url, params=params)
    time.sleep(SLEEP_PER_CALL)
    if not payload or "results" not in payload:
        df_empty = pd.DataFrame()
        _print_df(f"minute_aggregates_{symbol}_{from_dt}_{to_dt}", df_empty)
        return df_empty
    df = pd.DataFrame(payload["results"])
    if df.empty:
        _print_df(f"minute_aggregates_{symbol}_{from_dt}_{to_dt}", df)
        return df
    df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize(None)
    df.set_index("timestamp", inplace=True)
    _print_df(f"minute_aggregates_{symbol}_{from_dt}_{to_dt}", df)
    return df

def get_trades_for_date(symbol: str, date: str) -> pd.DataFrame:
    """Fetch trades for a symbol on a specific date using Polygon Trades endpoint.

    Returns a DataFrame with columns: timestamp, price, size, exchange, conditions, trade_id, trf_id (if present), side (if present or inferred).
    Date should be YYYY-MM-DD.
    """
    url = f"{BASE}/v2/ticks/stocks/trades/{symbol}/{date}"
    params = {"limit": 50000, "sort": "asc"}
    payload = _get(url, params=params)
    time.sleep(SLEEP_PER_CALL)
    if not payload or "results" not in payload:
        if payload is None:
            print(f"[polygon_io_basics] get_trades_for_date: no payload for {symbol} {date}")
        else:
            print(f"[polygon_io_basics] get_trades_for_date: unexpected payload for {symbol} {date}: {payload}")
        df_empty = pd.DataFrame()
        _print_df(f"trades_{symbol}_{date}", df_empty)
        return df_empty
    results = payload.get("results", [])
    if not results:
        df_empty = pd.DataFrame()
        _print_df(f"trades_{symbol}_{date}", df_empty)
        return df_empty
    # Normalize into DataFrame
    rows = []
    for r in results:
        # Polygon trade fields vary; handle common keys
        ts = r.get("t") or r.get("timestamp")
        price = r.get("p") or r.get("price")
        size = r.get("s") or r.get("size")
        exch = r.get("x") or r.get("exchange")
        cond = r.get("c") or r.get("conditions")
        trade_id = r.get("i") or r.get("trade_id")
        trf = r.get("trf_id") or r.get("trfId") or r.get("trf")
        # Build row
        rows.append({
            "timestamp": pd.to_datetime(ts, unit="ms") if ts is not None else None,
            "price": float(price) if price is not None else None,
            "size": int(size) if size is not None else None,
            "exchange": int(exch) if exch is not None else None,
            "conditions": cond,
            "trade_id": trade_id,
            "trf_id": trf,
            # keep raw payload for debugging if needed
            "raw": r,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.set_index("timestamp", inplace=True)
    # Infer side if possible: look for explicit side fields
    if "side" in df.columns:
        df["side"] = df["side"].astype(str)
    else:
        # try to infer from raw payload keys
        def infer_side_from_raw(raw):
            # polygon sometimes includes 'buyer_side' or 'seller_side' keys inside raw
            if isinstance(raw, dict):
                if raw.get("buyer_side"):
                    return raw.get("buyer_side")
                if raw.get("side"):
                    return raw.get("side")
            return None
        df["side"] = df["raw"].apply(infer_side_from_raw)
        # If still None, apply tick rule: compare to previous price
        if df["side"].isna().all():
            prices = df["price"].fillna(method="ffill")
            prev = prices.shift(1)
            tick = prices - prev
            df.loc[tick > 0, "side"] = "buy"
            df.loc[tick < 0, "side"] = "sell"
            df.loc[tick == 0, "side"] = "unknown"
    # Convert exchange and trf_id presence
    df["is_dark_pool"] = df.apply(lambda row: (row.get("exchange") == 4) and (row.get("trf_id") not in (None, "", [])), axis=1)
    _print_df(f"trades_{symbol}_{date}", df)
    return df

def summarize_dark_pool_trades(symbol: str, date: str) -> pd.DataFrame:
    """Return aggregated dark-pool trade stats for a symbol/date.

    Columns: symbol, date, dark_trades_count, dark_volume, dark_avg_size,
             buy_trades_count, buy_volume, sell_trades_count, sell_volume
    """
    df = get_trades_for_date(symbol, date)
    if df.empty:
        out_df = pd.DataFrame([{
            "symbol": symbol,
            "date": date,
            "dark_trades_count": 0,
            "dark_volume": 0,
            "dark_avg_size": 0,
            "buy_trades_count": 0,
            "buy_volume": 0,
            "sell_trades_count": 0,
            "sell_volume": 0,
        }])
        _print_df(f"dark_summary_{symbol}_{date}", out_df)
        return out_df
    dark = df[df["is_dark_pool"]]
    dark_trades_count = int(len(dark))
    dark_volume = int(dark["size"].sum()) if not dark.empty else 0
    dark_avg_size = float(dark["size"].mean()) if not dark.empty else 0.0
    buy = dark[dark["side"] == "buy"]
    sell = dark[dark["side"] == "sell"]
    buy_trades_count = int(len(buy))
    buy_volume = int(buy["size"].sum()) if not buy.empty else 0
    sell_trades_count = int(len(sell))
    sell_volume = int(sell["size"].sum()) if not sell.empty else 0
    return pd.DataFrame([{
        "symbol": symbol,
        "date": date,
        "dark_trades_count": dark_trades_count,
        "dark_volume": dark_volume,
        "dark_avg_size": dark_avg_size,
        "buy_trades_count": buy_trades_count,
        "buy_volume": buy_volume,
        "sell_trades_count": sell_trades_count,
        "sell_volume": sell_volume,
    }])

def get_reference(symbol: str) -> Dict:
    url = f"{BASE}/v3/reference/tickers/{symbol}"
    payload = _get(url)
    time.sleep(SLEEP_PER_CALL)
    return payload or {}

def get_corporate_actions(symbol: str, from_date: str, to_date: str) -> Dict:
    url = f"{BASE}/v3/reference/corporate-actions"
    params = {"ticker": symbol, "from": from_date, "to": to_date}
    payload = _get(url, params=params)
    time.sleep(SLEEP_PER_CALL)
    return payload or {}

# --- Technical indicators (simple implementations) ---
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

# --- Signal logic ---
def compute_signals(df: pd.DataFrame, symbol: Optional[str] = None, verbose: bool = False) -> Dict:
    """More robust signal computation.

    - Requires at least 50 days for reliable SMA(50). Falls back to 20-day check if 50 not available.
    - Detects cross in the last N days (default 5) instead of only checking the last row.
    - Returns diagnostics when verbose=True.
    """
    out = {"signal": "HOLD", "reason": "", "suggested_buy_time": None, "suggested_sell_time": None}
    if df.empty or len(df) < 20:
        out["reason"] = "insufficient data"
        return out

    close = df["close"].astype(float).copy()
    # Use stricter min_periods so SMAs are meaningful
    sma_short = close.rolling(window=20, min_periods=20).mean()
    sma_long = close.rolling(window=50, min_periods=50).mean()

    # If 50-period SMA is unavailable (too short history), compute with available data but mark it
    use_50 = sma_long.notna().sum() >= 2

    rsi_series = rsi(close)
    last_rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else float('nan')

    # Current price and pct change
    cur_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    today_pct = (cur_price - prev_price) / (prev_price + 1e-9) * 100.0

    # Prepare diff series and detect sign changes (crosses)
    if use_50:
        diff = sma_short - sma_long
    else:
        # fallback: compare sma_short to a longer-term ema if sma_long missing
        ema_long = close.ewm(span=50, adjust=False).mean()
        diff = sma_short - ema_long

    # sign of diff (NaNs -> 0)
    diff_sign = np.sign(diff.fillna(0))
    diff_sign_prev = diff_sign.shift(1).fillna(0)
    cross = (diff_sign_prev <= 0) & (diff_sign > 0)
    death_cross = (diff_sign_prev >= 0) & (diff_sign < 0)

    # check last N days for cross
    N = 5
    recent_cross = cross[-N:].any()
    recent_death = death_cross[-N:].any()

    # diagnostic values (last available)
    cur_short = float(sma_short.iloc[-1]) if not np.isnan(sma_short.iloc[-1]) else None
    cur_long = float(sma_long.iloc[-1]) if use_50 and not np.isnan(sma_long.iloc[-1]) else None
    prev_short = float(sma_short.iloc[-2]) if not np.isnan(sma_short.iloc[-2]) else None
    prev_long = float(sma_long.iloc[-2]) if use_50 and not np.isnan(sma_long.iloc[-2]) else None

    # --- Dark-pool proxy: analyze large-minute volume spikes on last trading day
    dark_pool_score = None
    dark_pool_price_dir = None
    try:
        if symbol is not None:
            # attempt to fetch minute aggregates for the last date in df
            last_date = df.index[-1].date().isoformat()
            min_df = get_minute_aggregates(symbol, last_date, last_date)
            if not min_df.empty:
                med = float(min_df['volume'].median()) if not min_df['volume'].empty else 0.0
                if med > 0:
                    large_mask = min_df['volume'] > (5 * med)
                    large_vol = float(min_df.loc[large_mask, 'volume'].sum())
                    total_vol = float(min_df['volume'].sum())
                    dark_pool_score = large_vol / (total_vol + 1e-9)
                    # price direction during large-volume minutes (positive => net buying pressure)
                    if large_mask.any():
                        pdeltas = (min_df.loc[large_mask, 'close'] - min_df.loc[large_mask, 'open']) / (min_df.loc[large_mask, 'open'] + 1e-9)
                        dark_pool_price_dir = float(pdeltas.mean())
                    else:
                        dark_pool_price_dir = 0.0
                else:
                    dark_pool_score = 0.0
                    dark_pool_price_dir = 0.0
            else:
                dark_pool_score = 0.0
                dark_pool_price_dir = 0.0
    except Exception as e:
        if verbose:
            print(f"[compute_signals] dark-pool proxy fetch error for {symbol}: {e}")
        dark_pool_score = 0.0
        dark_pool_price_dir = 0.0

    if verbose:
        print(f"[compute_signals] last_price={cur_price:.2f} today_pct={today_pct:.2f} last_rsi={last_rsi:.2f}")
        print(f"[compute_signals] prev_short={prev_short} prev_long={prev_long} cur_short={cur_short} cur_long={cur_long}")
        print(f"[compute_signals] recent_cross={recent_cross} recent_death={recent_death} use_50={use_50}")

    # Buy if golden cross occurred in last N days and RSI not extreme
    if recent_cross and (np.isnan(last_rsi) or last_rsi < 70):
        out["signal"] = "BUY"
        reason_parts = [f"recent golden-cross in last {N} days", f"RSI({last_rsi:.1f})"]
        # if dark-pool buying proxy present, strengthen buy
        if dark_pool_score is not None and dark_pool_score > 0.15 and dark_pool_price_dir is not None and dark_pool_price_dir > 0:
            reason_parts.append(f"dark-pool proxy strong ({dark_pool_score:.2f}) with net buying")
        out["reason"] = " + ".join(reason_parts)
        out["suggested_buy_time"] = next_market_open()
        out["suggested_sell_time"] = out["suggested_buy_time"] + timedelta(days=7)
    # Sell if death cross in last N days or RSI very high
    elif recent_death or (not np.isnan(last_rsi) and last_rsi > 80):
        out["signal"] = "SELL"
        reason_parts = [f"recent death-cross in last {N} days" if recent_death else f"RSI({last_rsi:.1f}) high"]
        # if dark-pool shows heavy selling pressure, include it
        if dark_pool_score is not None and dark_pool_score > 0.15 and dark_pool_price_dir is not None and dark_pool_price_dir < 0:
            reason_parts.append(f"dark-pool proxy heavy ({dark_pool_score:.2f}) with net selling")
        out["reason"] = " + ".join(reason_parts)
        out["suggested_sell_time"] = next_market_open()
    else:
        out["signal"] = "HOLD"
        out["reason"] = f"no clear cross in last {N} days; RSI {last_rsi:.1f}; today's % {today_pct:.2f}"

    out.update({
        "current_price": cur_price,
        "today_pct": round(today_pct, 2),
        "sma20": round(cur_short, 4) if cur_short is not None else None,
        "sma50": round(cur_long, 4) if cur_long is not None else None,
        "rsi": round(last_rsi, 2) if not np.isnan(last_rsi) else None,
        "dark_pool_score": round(dark_pool_score, 4) if dark_pool_score is not None else None,
        "dark_pool_price_dir": round(dark_pool_price_dir, 6) if dark_pool_price_dir is not None else None,
    })
    return out

def next_market_open(now: Optional[datetime] = None) -> datetime:
    """
    Returns next US market open datetime (naive, local). Market opens 09:30 Monday-Friday.
    This is a simple heuristic — does not account for holidays.
    """
    now = now or datetime.utcnow()
    # convert UTC to US/Eastern naive approx by offset (assume -5 or -4). Keep as UTC-based appointment.
    # For simplicity return next weekday at 09:30 UTC-converted approx; user should adjust to desired TZ.
    next_day = now
    while True:
        next_day = next_day + timedelta(days=1)
        if next_day.weekday() < 5:  # Mon-Fri
            return datetime.combine(next_day.date(), dtime(hour=13, minute=30))  # 09:30 ET ~ 13:30 UTC (approx)

# --- Main orchestration ---
def analyze_symbols(symbols: List[str]) -> pd.DataFrame:
    """
    For each symbol: fetch 2 years daily data, compute indicators and signal.
    Returns a DataFrame of results.
    """
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=730)
    from_s = from_date.isoformat()
    to_s = to_date.isoformat()

    rows = []
    for symbol in symbols:
        try:
            df = get_daily_aggregates(symbol, from_s, to_s)
            # compute_signals is deliberately not called here unless requested by caller
            ref = get_reference(symbol)
            rows.append({
                "symbol": symbol,
                "name": ref.get("results", {}).get("name") if ref.get("results") else None,
                "market": ref.get("results", {}).get("market") if ref.get("results") else None,
                "has_data": not df.empty,
                "data_points": len(df),
            })
        except Exception as e:
            rows.append({"symbol": symbol, "error": str(e)})
        # Respect rate limit
        time.sleep(SLEEP_PER_CALL)
    return pd.DataFrame(rows)


def get_last_n_days_hourly_summary(symbol: str, n: int = 5) -> Dict[str, pd.DataFrame]:
    """Return hourly volume DataFrame for last n trading days and dark-pool pre/post-market summary.

    Returns dict with keys: 'hourly_vol' (DataFrame with columns: date, hour_start, volume),
    and 'dark_prepost' (DataFrame with per-date dark pool counts/volumes before and after market).
    """
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=180)
    df = get_daily_aggregates(symbol, from_date.isoformat(), to_date.isoformat())
    if df.empty:
        print(f"[get_last_n_days_hourly_summary] No daily data for {symbol}")
        return {"hourly_vol": pd.DataFrame(), "dark_prepost": pd.DataFrame()}

    # get last n unique trading dates
    trading_dates = list(pd.Series(df.index.normalize().unique()).sort_values())
    if not trading_dates:
        return {"hourly_vol": pd.DataFrame(), "dark_prepost": pd.DataFrame()}
    recent_dates = trading_dates[-n:]

    hourly_rows = []
    dark_rows = []

    # Market hours in UTC approx: 13:30 - 20:00
    market_open = dtime(hour=13, minute=30)
    market_close = dtime(hour=20, minute=0)

    for dt in recent_dates:
        date_str = dt.date().isoformat()
        # minute-level aggregates for the day
        min_df = get_minute_aggregates(symbol, date_str, date_str)
        time.sleep(SLEEP_PER_CALL)
        if min_df.empty:
            # still append empty placeholders
            dark_rows.append({
                "symbol": symbol,
                "date": date_str,
                "dark_before_count": 0,
                "dark_before_volume": 0,
                "dark_after_count": 0,
                "dark_after_volume": 0,
            })
            continue

        # resample minute volumes to hourly buckets (on the timestamp index)
        # ensure timestamp index is sorted
        min_df = min_df.sort_index()
        # use pandas resample with label left (hour start)
        hourly = min_df["volume"].resample("60min", label="left", closed="left").sum()
        for ts, vol in hourly.iteritems():
            hourly_rows.append({
                "symbol": symbol,
                "date": ts.date().isoformat(),
                "hour_start": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "volume": int(vol),
            })

        # get trades to identify dark-pool trades (exchange==4 and trf_id present)
        trades_df = get_trades_for_date(symbol, date_str)
        time.sleep(SLEEP_PER_CALL)
        if trades_df.empty:
            dark_rows.append({
                "symbol": symbol,
                "date": date_str,
                "dark_before_count": 0,
                "dark_before_volume": 0,
                "dark_after_count": 0,
                "dark_after_volume": 0,
            })
            continue

        # classify trades as before market, during market, after market using UTC times
        trades_df = trades_df.sort_index()
        before_mask = trades_df.index.time < market_open
        after_mask = trades_df.index.time >= market_close
        before_dark = trades_df[before_mask & trades_df["is_dark_pool"]]
        after_dark = trades_df[after_mask & trades_df["is_dark_pool"]]

        dark_rows.append({
            "symbol": symbol,
            "date": date_str,
            "dark_before_count": int(len(before_dark)),
            "dark_before_volume": int(before_dark["size"].sum()) if not before_dark.empty else 0,
            "dark_after_count": int(len(after_dark)),
            "dark_after_volume": int(after_dark["size"].sum()) if not after_dark.empty else 0,
        })

    hourly_vol_df = pd.DataFrame(hourly_rows)
    dark_prepost_df = pd.DataFrame(dark_rows)
    _print_df(f"hourly_vol_{symbol}", hourly_vol_df)
    _print_df(f"dark_prepost_{symbol}", dark_prepost_df)
    return {"hourly_vol": hourly_vol_df, "dark_prepost": dark_prepost_df}


def _to_2023_date(d: datetime.date) -> datetime.date:
    """Map a date to the closest valid 2023 date preserving month/day when possible.
    If the exact month/day doesn't exist (e.g., Feb 29), step backward until a valid date is found.
    """
    y = 2023
    try:
        return d.replace(year=y)
    except Exception:
        # fallback: decrement day until valid
        dt = d
        while True:
            dt = dt - timedelta(days=1)
            try:
                return dt.replace(year=y)
            except Exception:
                continue


def compare_recent_with_2023(symbol: str, n: int = 5) -> None:
    """Fetch recent n business days and corresponding dates in 2023, print dark-pool summaries and prices.

    Prints two DataFrames: recent period and 2023 comparison.
    """
    # fetch last 180 days to ensure we have enough business days
    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=180)
    df = get_daily_aggregates(symbol, from_date.isoformat(), to_date.isoformat())
    if df.empty:
        print(f"[compare] No daily data for {symbol} in range {from_date}..{to_date}")
        return
    # get last n trading days
    recent_dates = list(df.index.normalize().unique()[-n:])
    recent_rows = []
    comp_rows = []
    for dt in recent_dates:
        date_str = dt.date().isoformat()
        close = df.loc[dt, 'close'] if dt in df.index else df[df.index.normalize() == dt].iloc[-1]['close']
        # dark pool summary for this date
        dark_df = summarize_dark_pool_trades(symbol, date_str)
        # extract values
        dr = dark_df.iloc[0].to_dict() if not dark_df.empty else {}
        recent_rows.append({
            'symbol': symbol,
            'date': date_str,
            'close': float(close) if not pd.isna(close) else None,
            'dark_trades_count': dr.get('dark_trades_count', 0),
            'dark_volume': dr.get('dark_volume', 0),
            'buy_volume': dr.get('buy_volume', 0),
            'sell_volume': dr.get('sell_volume', 0),
        })
        time.sleep(SLEEP_PER_CALL)

        # corresponding 2023 date
        d2023 = _to_2023_date(dt.date())
        d2023_str = d2023.isoformat()
        # try get close price for 2023 date
        df23 = get_daily_aggregates(symbol, d2023_str, d2023_str)
        close23 = None
        if not df23.empty:
            try:
                close23 = float(df23['close'].iloc[-1])
            except Exception:
                close23 = None
        dark23_df = summarize_dark_pool_trades(symbol, d2023_str)
        d23 = dark23_df.iloc[0].to_dict() if not dark23_df.empty else {}
        comp_rows.append({
            'symbol': symbol,
            'date_2025': date_str,
            'close_2025': float(close) if not pd.isna(close) else None,
            'dark_trades_count_2025': dr.get('dark_trades_count', 0),
            'dark_volume_2025': dr.get('dark_volume', 0),
            'date_2023': d2023_str,
            'close_2023': close23,
            'dark_trades_count_2023': d23.get('dark_trades_count', 0),
            'dark_volume_2023': d23.get('dark_volume', 0),
        })
        time.sleep(SLEEP_PER_CALL)

    recent_df = pd.DataFrame(recent_rows)
    comp_df = pd.DataFrame(comp_rows)
    _print_df(f"recent_5days_{symbol}", recent_df)
    _print_df(f"compare_5days_{symbol}_2023", comp_df)


# --- CLI usage ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Polygon-based buy/sell signal scanner (conservative rate).")
    parser.add_argument("--symbols", nargs="*", help="Symbols to analyze", default=[])
    parser.add_argument("--limit", type=int, help="Number of tickers to fetch from polygon reference", default=0)
    args = parser.parse_args()

    if not POLYGON_KEY:
        raise SystemExit("Set POLYGON_API_KEY environment variable before running.")

    if args.symbols:
        symbols = args.symbols
    elif args.limit and args.limit > 0:
        symbols = get_all_tickers(limit=args.limit)
    else:
        # example default watchlist
        symbols = ["AAPL", "MSFT", "TSLA", "AMD", "UAMY"]

    result_df = analyze_symbols(symbols)
    pd.set_option("display.max_columns", None)
    print(result_df.to_string(index=False))