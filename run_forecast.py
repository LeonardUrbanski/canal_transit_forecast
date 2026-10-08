"""
run_forecast.py - daily: forecast the open Kalshi weeks, compare with live prices, log both.

Run after collect_daily.py. Writes:
  forecast_log.csv               one row per strike per run (model + market)
  reports/forecast_<date>.txt    the printed summary
"""

import contextlib
import glob
import io
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import forecast as f
import kalshi

EDGE_MARGIN = 0.05        # model must beat the price by this much to flag it
STALE_DAYS = 2            # warn if the newest daily report is older than this


def alerts(now):
    out = []
    # Stale report data
    d = pd.read_csv("dcr_parsed.csv", parse_dates=["report_date"])
    age = (now.normalize() - d["report_date"].max()).days
    if age > STALE_DAYS:
        out.append(f"STALE: newest daily report is {age} days old ({d['report_date'].max().date()})")
    # PortWatch revision: did the two newest archived versions disagree on overlapping days?
    files = sorted(glob.glob("portwatch/panama_*_thru_*.csv"))
    if len(files) >= 2:
        a = f.load_portwatch(files[-2], glitch=None)
        b = f.load_portwatch(files[-1], glitch=None)
        diff = (b - a).dropna()
        changed = diff[diff != 0]
        if len(changed):
            out.append(f"PORTWATCH REVISION: {len(changed)} past days changed "
                       f"(net {changed.sum():+.0f}) between the last two versions")
    return out


def compare(result, prices):
    probs = result["sim"]["probs"]
    w = prices[prices["week_end"] == result["week_end"]].set_index("strike")
    rows = []
    for k, p in probs.items():
        if k not in w.index:
            rows.append({"strike": k, "model": p})
            continue
        m = w.loc[k]
        flag = ""
        if m["usable"]:
            if p > m["yes_ask"] + EDGE_MARGIN:
                flag = "model > ask: YES looks cheap"
            elif p < m["yes_bid"] - EDGE_MARGIN:
                flag = "model < bid: NO looks cheap"
        else:
            flag = "spread too wide"
        rows.append({"strike": k, "model": p, "bid": m["yes_bid"], "ask": m["yes_ask"],
                     "mid": m["mid"], "flag": flag})
    return pd.DataFrame(rows)


def main():
    now = pd.Timestamp.utcnow().tz_localize(None)
    try:
        prices = kalshi.open_markets()
    except Exception as e:
        print(f"Kalshi prices unavailable ({e}); forecasting without them")
        prices = pd.DataFrame(columns=["week_end", "strike", "usable"])

    weeks = sorted(prices["week_end"].unique()) if not prices.empty else []
    if not weeks:                                          # nothing listed: forecast this week anyway
        weeks = [now.normalize() + pd.Timedelta(days=(6 - now.dayofweek) % 7)]

    warnings = alerts(now)
    for msg in warnings:
        print("!!", msg)

    results = []
    for week_end in weeks:
        print("\n" + "=" * 70)
        listed = sorted(prices.loc[prices["week_end"] == week_end, "strike"].dropna().unique()) \
            if not prices.empty else []
        if listed:                                   # forecast exactly the strikes Kalshi lists
            f.SETTINGS["strikes"] = [float(k) for k in listed]
        result = f.run_forecast(week_end, now)
        results.append(result)
        mp = kalshi.market_probs(prices, week_end) if not prices.empty else {}
        f.log_forecast(result, market_probs=mp)
        table = compare(result, prices) if not prices.empty else pd.DataFrame()
        if not table.empty:
            print("\nModel vs market:")
            print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("Maintenance")
    try:
        f.score_forecasts()
    except Exception as e:
        print(f"Scoring failed: {e}")
    try:
        f.maybe_update_estimations()
    except Exception as e:
        print(f"Estimation update failed: {e}")

    if results:
        try:
            import dashboard
            regime_warn = [f"Week ending {r['week_end'].date()}: regime {r['regime']['regime']}"
                           for r in results if r["regime"]["regime"] != 1]
            path = dashboard.build_dashboard(results, prices, alerts=warnings + regime_warn)
            print(f"\nDashboard saved to {os.path.abspath(path)}")
            print(f"Latest copy: {os.path.abspath('reports/dashboard_latest.html')}")
        except Exception as e:
            print(f"\nDashboard failed: {e}")


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            main()
    finally:                                   # print and save even if something failed
        text = buf.getvalue()
        print(text)
        stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H%M")
        with open(f"reports/forecast_{stamp}.txt", "w", encoding="utf-8") as fh:
            fh.write(text)
