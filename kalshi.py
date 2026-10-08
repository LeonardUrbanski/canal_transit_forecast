"""
kalshi.py - Kalshi market data for the Panama weekly series (public endpoints, no key).

- open_markets():        live strikes for listed weeks with bid/ask/mid
- save_price_snapshot(): append current prices to kalshi_prices.csv (price history)
- update_settled():      keep kalshi_settled.csv current with results
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "<Kalshi series ticker>"
HEADERS = {"User-Agent": "canal-traffic-research script"}
PRICES_CSV = "kalshi_prices.csv"
SETTLED_CSV = "kalshi_settled.csv"
MAX_SPREAD = 0.20          # wider than this: quotes too thin to treat the mid as a price


def _get(path, params=None, retries=4):
    for attempt in range(retries):
        r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi request failed: {path}")


def _markets(status):
    out, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data = _get("/markets", params)
        out += data.get("markets", [])
        cursor = data.get("cursor")
        if not cursor:
            return out


def _price(m, name):
    """Price in dollars (0-1). Handles both '<name>_dollars' strings and integer cents."""
    v = m.get(f"{name}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(name)
    return None if v is None else float(v) / 100


def _week_end(event_ticker):
    return pd.to_datetime(event_ticker.split("-")[1], format="%y%b%d")


def _rows(markets):
    rows = []
    for m in markets:
        bid, ask = _price(m, "yes_bid"), _price(m, "yes_ask")
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        spread = (ask - bid) if mid is not None else None
        rows.append({
            "ticker": m["ticker"], "event_ticker": m["event_ticker"],
            "week_end": _week_end(m["event_ticker"]),
            "strike": float(m["floor_strike"]) if m.get("floor_strike") is not None else None,
            "strike_type": m.get("strike_type"),
            "yes_bid": bid, "yes_ask": ask, "mid": mid, "spread": spread,
            "usable": mid is not None and spread is not None and spread <= MAX_SPREAD,
            "last_price": _price(m, "last_price"),
            "volume": m.get("volume_fp", m.get("volume")),
            "close_time": m.get("close_time"), "status": m.get("status"),
            "result": m.get("result"), "expiration_value": m.get("expiration_value"),
        })
    return pd.DataFrame(rows)


def open_markets():
    """Live markets with prices. Columns: week_end, strike, yes_bid, yes_ask, mid, spread, usable..."""
    df = _rows(_markets("open"))
    return df.sort_values(["week_end", "strike"]).reset_index(drop=True) if not df.empty else df


def market_probs(df, week_end):
    """{strike: mid} for one week, only where the quote is usable (tight enough spread)."""
    w = df[(df["week_end"] == pd.Timestamp(week_end)) & df["usable"]]
    return {float(k): round(float(v), 4) for k, v in zip(w["strike"], w["mid"])}


def save_price_snapshot(path=PRICES_CSV):
    df = open_markets()
    if df.empty:
        print("Kalshi: no open markets")
        return df
    df.insert(0, "captured_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    weeks = ", ".join(str(w.date()) for w in sorted(df["week_end"].unique()))
    print(f"Kalshi: saved prices for {len(df)} open markets (weeks ending {weeks})")
    return df


def update_settled(path=SETTLED_CSV):
    df = _rows(_markets("settled"))
    if df.empty:
        print("Kalshi: no settled markets returned")
        return df
    df = df[["ticker", "event_ticker", "week_end", "strike", "result", "expiration_value"]]
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["week_end"])
        df = pd.concat([old, df]).drop_duplicates("ticker", keep="last")
    df.sort_values(["week_end", "strike"]).to_csv(path, index=False)
    print(f"Kalshi: {df['event_ticker'].nunique()} settled weeks on file")
    return df
