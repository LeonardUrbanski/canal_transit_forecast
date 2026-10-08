"""
forecast.py - Panama Canal weekly transit forecast (regime 1: demand exceeds capacity).

Step 1: loaders  -> clean tables from the files collect_daily.py saves
Step 3: daily canal estimate -> for each day of a market week, the best
        available ACP transit estimate as of a given time, with its uncertainty

Conventions
- All "as_of" times are naive UTC timestamps (snapshot filenames are UTC).
- A daily report dated R contains the actual transits for R - 1 and schedules
  for R, R + 1, R + 2. It is treated as available from R onward.
- Booked totals = Panamax (regular feed) + Neopanamax (neo feed), summed over
  the BN/BS fields of each date's booking rows.

Placeholders that should be re-measured as data accumulates live in SETTINGS.
"""

import datetime as _dt
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings (update these as measurements come in)
# ---------------------------------------------------------------------------
SETTINGS = {
    # NOTE: the values below are generic placeholders, not the tuned values used for live
    # forecasts. See README "Parameters" for how to calibrate them from collected data.
    # Schedule error (std, transits/day) by horizon: 0 = today, 1 = tomorrow, 2 = next day.
    "schedule_sd": {0: 1.0, 1: 1.5, 2: 2.5},
    # Transits beyond booked slots (auction slots, unbooked vessels).
    "gap_mean": 0.0,
    "gap_sd": 2.0,
    # Bookings are trusted this many days ahead.
    "booking_horizon_days": 13,
    # Beyond the booking horizon: daily slot cap by effective date (published by the ACP),
    # plus how far actual transits run above the cap.
    "caps": [("2026-09-15", 32), ("2026-10-15", 33)],
    "cap_excess": 0.0,
    "cap_sd": 3.0,
    # --- Step 4: PortWatch / ACP ratio ---
    # Seasonal normal by calendar month. Placeholder: flat. Estimate from ACP monthly
    # summaries vs PortWatch history.
    "ratio_seasonal": {m: 0.93 for m in range(1, 13)},
    "ratio_halflife_weeks": 3.0,   # EWMA half-life for the weekly deviation
    "ratio_noise_default": 0.03,   # weekly ratio noise used until 3+ weeks are observed
    "ratio_sd_floor": 0.03,        # never claim more certainty than this
    "pw_release_lag_days": 2,      # PortWatch publishes a Mon-Sun week on the following Tuesday
    # --- Step 5: simulation ---
    "strikes": [175, 200, 225, 250, 275],
    "n_sims": 100_000,
    # Disruption days (weather, closures). Rate, size, and PortWatch passthrough are
    # placeholders; estimate from observed disruptions.
    "disruption_prob_per_day": 0.02,
    "disruption_shortfall": 10,          # ACP transits lost on a disruption day
    "disruption_pw_passthrough": 1.0,    # share of a disruption that shows up in PortWatch
    # --- Step 2: regime check ---
    "regime_window_days": (2, 13),       # dates checked for open slots, relative to as_of
    "model_version": "public-v1",
    # --- Naive EWMA benchmark / regime 2 fallback ---
    "ewma_halflife_weeks": 2.0,
    "ewma_error_years": 3,          # history used for the EWMA's error distribution
    # Weeks left out of the ratio estimate, with reasons
    "ratio_exclude_weeks": {},
}

# ---------------------------------------------------------------------------
# Estimations file: uncertainty numbers live in estimations.json, not in code
# ---------------------------------------------------------------------------
ESTIMATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimations.json")


def load_estimations(path=ESTIMATIONS_PATH, settings=SETTINGS):
    """Overwrite SETTINGS values with those in estimations.json (if present)."""
    if not os.path.exists(path):
        print(f"(no {os.path.basename(path)} found, using built-in defaults)")
        return settings
    with open(path, encoding="utf-8") as f:
        est = json.load(f)
    for key, entry in est.items():
        if key.startswith("_") or key not in settings and key != "disruption_days":
            continue
        value = entry["value"]
        if key == "schedule_sd":
            value = {int(k): v for k, v in value.items()}
        settings[key] = value
    return settings


def show_estimations(path=ESTIMATIONS_PATH):
    """Print every estimate with where it came from."""
    with open(path, encoding="utf-8") as f:
        est = json.load(f)
    rows = [{"name": k, "value": v["value"], "source": v.get("source"), "n": v.get("n"),
             "updated": v.get("updated"), "note": v.get("note", "")}
            for k, v in est.items() if not k.startswith("_")]
    print(pd.DataFrame(rows).to_string(index=False))


def update_estimations(dcr_path="dcr_parsed.csv", booking_dir="booking_snapshots",
                       path=ESTIMATIONS_PATH, min_days=5, write=True):
    """Re-measure what the data supports and save it to estimations.json.

    - schedule_sd: schedule vs actual by horizon (disruption days excluded)
    - gap_mean / gap_sd: actual minus booked, using the latest snapshot taken at least
      one day before each date (needs min_days of overlap, else left unchanged)
    Everything else stays as it is in the file."""
    with open(path, encoding="utf-8") as f:
        est = json.load(f)
    today = _dt.date.today().isoformat()
    skip = pd.to_datetime(est.get("disruption_days", {}).get("value", []))
    actuals, schedules = load_dcr(dcr_path)
    actuals = actuals.drop(index=skip, errors="ignore")

    # Schedule error by horizon
    sc = schedules.copy()
    sc["actual"] = sc["target_date"].map(actuals)
    sc = sc.dropna(subset=["actual"])
    sc["error"] = sc["actual"] - sc["scheduled"]
    sds, ns = {}, {}
    for h, g in sc.groupby("horizon"):
        if len(g) >= min_days:
            sds[str(h)], ns[str(h)] = round(float(g["error"].std(ddof=1)), 2), len(g)
    if sds:
        old = est["schedule_sd"]["value"]
        est["schedule_sd"] = {"value": {**old, **sds}, "source": "measured", "n": min(ns.values()),
                              "updated": today, "note": f"n by horizon {ns}; disruption days excluded"}
    print("schedule_sd:", est["schedule_sd"]["value"], "n:", ns)

    # Booking gap: actual - booked, from a snapshot taken >= 1 day before the date
    bookings = load_booking_history(booking_dir)
    rows = []
    for date, act in actuals.items():
        snaps = bookings[(bookings["date"] == date) &
                         (bookings["snapshot_time"].dt.normalize() < date)]
        if not snaps.empty:
            latest = snaps.sort_values("snapshot_time").iloc[-1]
            if latest["booked_total"] > 0:
                rows.append({"date": date, "booked": latest["booked_total"], "actual": act})
    gaps = pd.DataFrame(rows)
    if len(gaps) >= min_days:
        g = gaps["actual"] - gaps["booked"]
        est["gap_mean"] = {"value": round(float(g.mean()), 2), "source": "measured", "n": len(g),
                           "updated": today, "note": "actual minus booked (snapshot >= 1 day ahead)"}
        est["gap_sd"] = {"value": round(float(g.std(ddof=1)), 2), "source": "measured", "n": len(g),
                         "updated": today}
        print(f"gap: mean {g.mean():.2f}, sd {g.std(ddof=1):.2f}, n {len(g)}")
    else:
        print(f"gap: only {len(gaps)} day(s) of booked-vs-actual so far (need {min_days}); left unchanged")
    if not gaps.empty:
        print(gaps.assign(gap=gaps["actual"] - gaps["booked"]).to_string(index=False))

    if write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(est, f, indent=2)
        load_estimations(path)          # apply immediately
    return est


# ---------------------------------------------------------------------------
# Step 1: loaders
# ---------------------------------------------------------------------------

def load_dcr(path="dcr_parsed.csv"):
    """Daily reports -> (actuals, schedules).

    actuals:   Series indexed by actual_date -> actual ACP transits
    schedules: DataFrame with columns report_date, target_date, horizon, scheduled
    """
    d = pd.read_csv(path, parse_dates=["report_date", "actual_date"])
    d = d.drop_duplicates("report_date").sort_values("report_date")

    actuals = (d.dropna(subset=["actual_transits"])
                .set_index("actual_date")["actual_transits"].astype(float))

    rows = []
    for _, r in d.iterrows():
        for h, col in [(0, "sched_today"), (1, "sched_tomorrow"), (2, "sched_next")]:
            if pd.notna(r.get(col)):
                rows.append({"report_date": r["report_date"],
                             "target_date": r["report_date"] + pd.Timedelta(days=h),
                             "horizon": h, "scheduled": float(r[col])})
    schedules = pd.DataFrame(rows)
    return actuals, schedules


def _read_json(path):
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))


def _snapshot_time(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.json$", os.path.basename(path))
    if not m:
        raise ValueError(f"can't read timestamp from {path}")
    return pd.Timestamp(f"{m[1]} {m[2]}:{m[3]}")


def parse_booking_file(path, feed):
    """One booking snapshot -> DataFrame indexed by date: booked, available, period."""
    rows = []
    for rec in _read_json(path):
        bookings = rec.get("Bookings") or []
        booked = sum(b.get("BN", 0) + b.get("BS", 0) for b in bookings)
        # Open slots from the booking rows (DN/DS = available north/southbound).
        # The summary fields (Sup_ava, Reg_ava, Neo_ava) are not used: they can read 0
        # while the rows show open slots.
        if bookings:
            available = sum(max(b.get("DN", 0), 0) + max(b.get("DS", 0), 0) for b in bookings)
        else:
            available = np.nan        # no rows: unknown, don't guess
        rows.append({"date": pd.Timestamp(rec["bday"]), "booked": booked,
                     "available": available, "period": str(rec.get("period"))})
    return pd.DataFrame(rows).set_index("date")


def load_booking_history(folder="booking_snapshots"):
    """All snapshot pairs -> long table:
    snapshot_time, date, booked_panamax, booked_neo, booked_total,
    avail_panamax, avail_neo, period_panamax, period_neo
    """
    frames = []
    for reg_path in sorted(glob.glob(os.path.join(folder, "booking_regular_*.json"))):
        neo_path = reg_path.replace("booking_regular_", "booking_neo_")
        if not os.path.exists(neo_path):
            continue
        reg = parse_booking_file(reg_path, "regular")
        neo = parse_booking_file(neo_path, "neo")
        both = reg.join(neo, how="outer", lsuffix="_panamax", rsuffix="_neo")
        both = both.rename(columns={"booked_panamax": "booked_panamax", "booked_neo": "booked_neo",
                                    "available_panamax": "avail_panamax", "available_neo": "avail_neo"})
        for c in ["booked_panamax", "booked_neo"]:
            both[c] = both[c].fillna(0)
        both["booked_total"] = both["booked_panamax"] + both["booked_neo"]
        both["snapshot_time"] = _snapshot_time(reg_path)
        frames.append(both.reset_index())
    if not frames:
        raise FileNotFoundError(f"no booking snapshot pairs in {folder}")
    return pd.concat(frames, ignore_index=True)


PW_TYPES = ["container", "dry_bulk", "general_cargo", "roll-on/roll-off", "tanker"]

def _norm(s):
    return re.sub(r"[\s_]+", "_", s.strip().lower()).replace("roll_on/roll_off", "roll-on/roll-off")

def load_portwatch(path, glitch=("2020-01-26", "2020-02-02")):
    """PortWatch CSV -> daily totals (Series), Jan 2020 glitch days set to NaN."""
    pw = pd.read_csv(path)
    pw.columns = [_norm(c) for c in pw.columns]
    date_col = next(c for c in pw.columns if c in ("datetime", "date"))
    pw[date_col] = pd.to_datetime(pw[date_col])
    types = [c for c in PW_TYPES if c in pw.columns]
    if len(types) != 5:
        raise ValueError(f"expected 5 vessel-type columns, found {types}")
    daily = pw.set_index(date_col)[types].sum(axis=1).astype(float).sort_index()
    if glitch:
        daily.loc[glitch[0]:glitch[1]] = np.nan
    daily.name = "portwatch"
    return daily


def load_kalshi_markets(path="data/kalshi_markets.csv"):
    """Kalshi markets -> one row per market with week_end, strike, result."""
    mk = pd.read_csv(path)
    mk["week_end"] = pd.to_datetime(mk["event_ticker"].str.split("-").str[1], format="%y%b%d")
    mk["strike"] = mk["floor_strike"].astype(float)
    mk["hit"] = (mk["result"] == "yes").astype(int)
    return mk


# ---------------------------------------------------------------------------
# Step 3: daily canal estimate
# ---------------------------------------------------------------------------

def market_week(week_end):
    """Monday-Sunday dates for the market week ending on week_end (a Sunday)."""
    end = pd.Timestamp(week_end).normalize()
    if end.dayofweek != 6:
        raise ValueError(f"{end.date()} is not a Sunday")
    return pd.date_range(end - pd.Timedelta(days=6), end, freq="D")


def cap_for(date, settings=SETTINGS):
    cap = None
    for start, value in settings["caps"]:
        if pd.Timestamp(date) >= pd.Timestamp(start):
            cap = value
    if cap is None:
        raise ValueError(f"no cap defined for {date}")
    return cap


def daily_canal_estimate(dates, as_of, actuals, schedules, bookings, settings=SETTINGS):
    """Best ACP transit estimate for each date, using only information available at as_of.

    Returns DataFrame indexed by date: source, mean, sd, detail.
    Source priority: actual > schedule > bookings + gap > cap + gap.
    """
    as_of = pd.Timestamp(as_of)
    today = as_of.normalize()

    # Reports available by as_of (report dated R counts as available from day R)
    sched_av = schedules[schedules["report_date"] <= today]
    act_av = actuals[actuals.index + pd.Timedelta(days=1) <= today]

    # Latest booking snapshot taken before as_of
    snap_times = bookings["snapshot_time"].unique()
    snap_times = [t for t in snap_times if pd.Timestamp(t) <= as_of]
    snap = None
    if snap_times:
        latest = max(snap_times)
        snap = bookings[bookings["snapshot_time"] == latest].set_index("date")

    out = []
    for date in pd.DatetimeIndex(dates):
        days_ahead = (date - today).days

        if date in act_av.index:
            out.append((date, "actual", act_av[date], 0.0, ""))
            continue

        s = sched_av[sched_av["target_date"] == date]
        if not s.empty:
            row = s.sort_values("report_date").iloc[-1]          # most recent report
            h = int(row["horizon"])
            out.append((date, f"schedule_h{h}", row["scheduled"],
                        settings["schedule_sd"][h], f"report {row['report_date'].date()}"))
            continue

        if snap is not None and date in snap.index and days_ahead <= settings["booking_horizon_days"]:
            booked = snap.loc[date, "booked_total"]
            if booked > 0:
                out.append((date, "bookings", booked + settings["gap_mean"], settings["gap_sd"],
                            f"booked {int(booked)} @ {pd.Timestamp(latest)}"))
                continue

        cap = cap_for(date, settings)
        out.append((date, "cap", cap + settings["cap_excess"], settings["cap_sd"], f"cap {cap}"))

    return pd.DataFrame(out, columns=["date", "source", "mean", "sd", "detail"]).set_index("date")


def weekly_canal_estimate(week_end, as_of, actuals, schedules, bookings, settings=SETTINGS):
    """Daily table for the market week plus the weekly total (mean, sd), assuming
    independent daily errors. Disruption risk is handled later, in the simulation step."""
    daily = daily_canal_estimate(market_week(week_end), as_of, actuals, schedules, bookings, settings)
    total_mean = daily["mean"].sum()
    total_sd = float(np.sqrt((daily["sd"] ** 2).sum()))
    return daily, total_mean, total_sd


# ---------------------------------------------------------------------------
# Step 4: PortWatch / ACP ratio
# ---------------------------------------------------------------------------

def acp_daily_series(actuals, schedules):
    """Best available ACP count per day: actual, else the closest schedule
    (today, then tomorrow, then next-day horizon). Returns DataFrame: acp, source."""
    out = pd.DataFrame({"acp": actuals, "source": "actual"})
    if not schedules.empty:
        for h in (0, 1, 2):
            s = (schedules[schedules["horizon"] == h]
                 .sort_values("report_date").drop_duplicates("target_date", keep="last")
                 .set_index("target_date")["scheduled"])
            missing = s.index.difference(out.index)
            if len(missing):
                out = pd.concat([out, pd.DataFrame({"acp": s[missing], "source": f"schedule_h{h}"})])
    return out.sort_index()


def seasonal_ratio(date, settings=SETTINGS):
    """Seasonal normal for the week containing date (uses the Thursday's month)."""
    d = pd.Timestamp(date)
    thursday = d - pd.Timedelta(days=d.dayofweek) + pd.Timedelta(days=3)
    return settings["ratio_seasonal"][thursday.month]


def weekly_ratio_table(pw_daily, actuals, schedules, as_of=None, settings=SETTINGS):
    """Mon-Sun weeks where both PortWatch and ACP cover all 7 days.

    Only weeks PortWatch would have published by as_of are included
    (week_end + release lag <= as_of). Columns: acp, portwatch, ratio,
    seasonal, deviation, est_days (ACP days filled from schedules)."""
    acp = acp_daily_series(actuals, schedules)
    df = pd.DataFrame({"acp": acp["acp"], "est": (acp["source"] != "actual").astype(int),
                       "portwatch": pw_daily})
    df = df.dropna(subset=["acp", "portwatch"])
    if df.empty:
        return pd.DataFrame()
    g = df.resample("W-SUN")
    weeks = pd.DataFrame({"acp": g["acp"].sum(), "portwatch": g["portwatch"].sum(),
                          "days": g["acp"].count(), "est_days": g["est"].sum()})
    weeks = weeks[weeks["days"] == 7].drop(columns="days")
    if as_of is not None:
        published = weeks.index + pd.Timedelta(days=settings["pw_release_lag_days"])
        weeks = weeks[published <= pd.Timestamp(as_of)]
    weeks["ratio"] = weeks["portwatch"] / weeks["acp"]
    weeks["seasonal"] = [seasonal_ratio(w, settings) for w in weeks.index]
    weeks["deviation"] = weeks["ratio"] - weeks["seasonal"]
    return weeks


def ratio_estimate(target_week_end, as_of, pw_daily, actuals, schedules, settings=SETTINGS):
    """Expected PortWatch/ACP ratio for the target week, with uncertainty.

    ratio = seasonal normal for the target week + EWMA of recent weekly deviations.
    sd combines week-to-week noise with uncertainty in the estimated level."""
    weeks = weekly_ratio_table(pw_daily, actuals, schedules, as_of, settings)
    excluded = pd.to_datetime(list(settings.get("ratio_exclude_weeks", {})))
    if not weeks.empty:
        weeks = weeks.drop(index=excluded, errors="ignore")
    seasonal = seasonal_ratio(target_week_end, settings)
    if weeks.empty:
        sd = max(settings["ratio_noise_default"] * 2, settings["ratio_sd_floor"])
        return {"ratio": seasonal, "sd": sd, "seasonal": seasonal, "deviation": 0.0,
                "n_weeks": 0, "weeks": weeks}

    dev = weeks["deviation"]
    smoothed = dev.ewm(halflife=settings["ratio_halflife_weeks"]).mean().iloc[-1]

    n = len(dev)
    noise = dev.std(ddof=1) if n >= 3 else settings["ratio_noise_default"]
    w = 0.5 ** (np.arange(n)[::-1] / settings["ratio_halflife_weeks"])
    n_eff = w.sum() ** 2 / (w ** 2).sum()                     # effective number of weeks
    sd = max(np.sqrt(noise ** 2 + noise ** 2 / n_eff), settings["ratio_sd_floor"])

    return {"ratio": float(seasonal + smoothed), "sd": float(sd), "seasonal": seasonal,
            "deviation": float(smoothed), "n_weeks": n, "weeks": weeks}


# ---------------------------------------------------------------------------
# Step 2: regime check
# ---------------------------------------------------------------------------

def regime_status(bookings, as_of, settings=SETTINGS):
    """Regime 1 (demand exceeds capacity) if no near-term date has open slots
    in the latest snapshot before as_of. Returns dict: regime, open_slots (Series)."""
    as_of = pd.Timestamp(as_of)
    times = [t for t in bookings["snapshot_time"].unique() if pd.Timestamp(t) <= as_of]
    if not times:
        return {"regime": None, "open_slots": pd.Series(dtype=float), "snapshot": None}
    latest = max(times)
    snap = bookings[bookings["snapshot_time"] == latest].set_index("date")
    lo, hi = settings["regime_window_days"]
    start, end = as_of.normalize() + pd.Timedelta(days=lo), as_of.normalize() + pd.Timedelta(days=hi)
    window = snap.loc[start:end]
    open_slots = window["avail_panamax"] + window["avail_neo"]
    if open_slots.isna().any() or open_slots.empty:
        regime = None                 # missing data in the window: can't tell
    else:
        regime = 1 if (open_slots == 0).all() else 2
    return {"regime": regime, "open_slots": open_slots, "snapshot": pd.Timestamp(latest)}


# ---------------------------------------------------------------------------
# Step 5: probabilities
# ---------------------------------------------------------------------------

def simulate_week(daily, ratio, settings=SETTINGS, seed=0):
    """Monte Carlo of the PortWatch weekly total.

    daily: output of daily_canal_estimate (source, mean, sd per day)
    ratio: output of ratio_estimate (ratio, sd)
    Returns dict: probs {strike: P(total > strike)}, mean, sd, quantiles, sims."""
    rng = np.random.default_rng(seed)
    n = settings["n_sims"]
    means, sds = daily["mean"].to_numpy(), daily["sd"].to_numpy()
    canal = rng.normal(means, sds, size=(n, len(daily)))

    # Disruptions only on days not yet observed
    at_risk = (daily["source"] != "actual").to_numpy()
    hits = rng.random((n, len(daily))) < settings["disruption_prob_per_day"]
    shortfall = (hits & at_risk) * settings["disruption_shortfall"]
    canal_total = canal.sum(axis=1)
    pw_shortfall = shortfall.sum(axis=1) * settings["disruption_pw_passthrough"]

    r = rng.normal(ratio["ratio"], ratio["sd"], size=n)
    pw_total = np.round((canal_total - pw_shortfall) * r)

    probs = {k: float((pw_total > k).mean()) for k in settings["strikes"]}
    q = np.percentile(pw_total, [5, 25, 50, 75, 95])
    return {"probs": probs, "mean": float(pw_total.mean()), "sd": float(pw_total.std()),
            "quantiles": dict(zip([5, 25, 50, 75, 95], q.tolist())), "sims": pw_total}



# ---------------------------------------------------------------------------
# Naive benchmark: EWMA of PortWatch's own weekly totals
# ---------------------------------------------------------------------------

def pw_weekly(pw_daily):
    """Complete Mon-Sun PortWatch weekly totals (weeks with missing days dropped)."""
    g = pw_daily.resample("W-SUN")
    totals, days = g.sum(), g.count()
    return totals[days == 7]


def ewma_forecast(week_end, as_of, pw_daily, settings=SETTINGS):
    """Forecast a week's PortWatch total from PortWatch's own published history only.

    Probabilities come from the EWMA's past errors at the same horizon (weeks between
    the last published week and the target week). Returns None if nothing is published."""
    weekly = pw_weekly(pw_daily)
    published = weekly[weekly.index + pd.Timedelta(days=settings["pw_release_lag_days"])
                       <= pd.Timestamp(as_of)]
    if len(published) < 30:
        return None
    week_end = pd.Timestamp(week_end)
    last = published.index[-1]
    h = max(1, (week_end - last).days // 7)
    ew = published.ewm(halflife=settings["ewma_halflife_weeks"]).mean()
    point = float(ew.iloc[-1])

    # Past errors at horizon h: actual week minus the EWMA available h weeks earlier
    shifted = ew.shift(freq=f"{7 * h}D")
    errors = (published - shifted).dropna()
    errors = errors[errors.index >= last - pd.DateOffset(years=settings["ewma_error_years"])]
    sims = np.round(point + errors.to_numpy())
    probs = {k: float((sims > k).mean()) for k in settings["strikes"]}
    return {"mean": float(sims.mean()), "sd": float(sims.std()), "point": point, "probs": probs,
            "horizon_weeks": h, "last_week": last, "n_errors": len(errors)}

def run_forecast(week_end, as_of=None, dcr_path="dcr_parsed.csv", booking_dir="booking_snapshots",
                 pw_path="portwatch_panama_latest.csv", settings=SETTINGS, verbose=True):
    """Full pipeline for one market week. Returns a dict with every intermediate result."""
    as_of = pd.Timestamp.utcnow().tz_localize(None) if as_of is None else pd.Timestamp(as_of)
    actuals, schedules = load_dcr(dcr_path)
    bookings = load_booking_history(booking_dir)
    pw = load_portwatch(pw_path)

    regime = regime_status(bookings, as_of, settings)
    daily, canal_mean, canal_sd = weekly_canal_estimate(week_end, as_of, actuals, schedules, bookings, settings)
    ratio = ratio_estimate(week_end, as_of, pw, actuals, schedules, settings)
    sim = simulate_week(daily, ratio, settings)
    ewma = ewma_forecast(week_end, as_of, pw, settings)
    # Regime 1: the structural model. Otherwise (open slots or unknown) fall back to the EWMA.
    primary = "structural" if regime["regime"] == 1 or ewma is None else "ewma"

    result = {"week_end": pd.Timestamp(week_end), "as_of": as_of, "regime": regime,
              "daily": daily, "canal_mean": canal_mean, "canal_sd": canal_sd,
              "ratio": ratio, "sim": sim, "ewma": ewma, "primary": primary}
    if verbose:
        print(f"Market week ending {pd.Timestamp(week_end).date()}, as of {as_of:%Y-%m-%d %H:%M} UTC")
        if regime["regime"] != 1:
            print(f"WARNING: regime {regime['regime']}: open slots found\n{regime['open_slots'][regime['open_slots'] > 0]}")
        print(daily.to_string())
        print(f"Canal (ACP):  {canal_mean:.0f} +/- {canal_sd:.1f}")
        print(f"Ratio:        {ratio['ratio']:.3f} +/- {ratio['sd']:.3f} "
              f"(seasonal {ratio['seasonal']:.3f}, deviation {ratio['deviation']:+.3f}, {ratio['n_weeks']} weeks)")
        print(f"PortWatch:    {sim['mean']:.0f} +/- {sim['sd']:.1f}  "
              f"(90% range {sim['quantiles'][5]:.0f}-{sim['quantiles'][95]:.0f})")
        for k, p in sim["probs"].items():
            e = f"   (EWMA {ewma['probs'][k]:6.1%})" if ewma else ""
            print(f"  P(above {k}) = {p:6.1%}{e}")
        if ewma:
            print(f"EWMA benchmark: {ewma['mean']:.0f} +/- {ewma['sd']:.1f} "
                  f"(horizon {ewma['horizon_weeks']} wk, last published week {ewma['last_week'].date()})")
        if primary == "ewma":
            print("FALLBACK: regime is not 1, so the EWMA probabilities are logged as the primary forecast")
    return result


# ---------------------------------------------------------------------------
# Step 6: forecast log and scoring
# ---------------------------------------------------------------------------

LOG_FIELDS = ["logged_at", "week_end", "as_of", "strike", "prob", "market_prob",
              "pw_mean", "pw_sd", "canal_mean", "ratio", "ratio_sd", "regime", "model_version",
              "prob_structural", "prob_ewma", "ewma_mean", "ewma_sd", "primary"]


def _migrate_log(path):
    """Add any new columns to an existing log so appends line up."""
    if not os.path.exists(path):
        return
    old = pd.read_csv(path)
    missing = [c for c in LOG_FIELDS if c not in old.columns]
    if missing:
        for c in missing:
            old[c] = np.nan
        old[LOG_FIELDS].to_csv(path, index=False)


def log_forecast(result, path="forecast_log.csv", market_probs=None, settings=SETTINGS):
    """Append one row per strike. market_probs: optional {strike: price 0-1} at the same time,
    e.g. the Kalshi mid price you see when you run the forecast."""
    market_probs = market_probs or {}
    _migrate_log(path)
    now = pd.Timestamp.utcnow().tz_localize(None)
    ewma = result.get("ewma")
    primary = result.get("primary", "structural")
    rows = [{"logged_at": now, "week_end": result["week_end"].date(), "as_of": result["as_of"],
             "strike": k,
             "prob": round(ewma["probs"][k] if primary == "ewma" else p, 4),
             "prob_structural": round(p, 4),
             "prob_ewma": round(ewma["probs"][k], 4) if ewma else None,
             "ewma_mean": round(ewma["mean"], 1) if ewma else None,
             "ewma_sd": round(ewma["sd"], 2) if ewma else None,
             "primary": primary,
             "market_prob": market_probs.get(k),
             "pw_mean": round(result["sim"]["mean"], 1), "pw_sd": round(result["sim"]["sd"], 2),
             "canal_mean": round(result["canal_mean"], 1), "ratio": round(result["ratio"]["ratio"], 4),
             "ratio_sd": round(result["ratio"]["sd"], 4), "regime": result["regime"]["regime"],
             "model_version": settings["model_version"]}
            for k, p in result["sim"]["probs"].items()]
    df = pd.DataFrame(rows, columns=LOG_FIELDS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return df


def first_release_weekly(pw_dir="portwatch"):
    """Weekly PortWatch totals as FIRST published (what Kalshi settles on), from the
    archived versions collect_daily.py saves. Returns Series indexed by week_end."""
    files = sorted(glob.glob(os.path.join(pw_dir, "panama_*_thru_*.csv")))
    first = {}
    for path in files:                                   # oldest version first
        daily = load_portwatch(path, glitch=None)
        g = daily.resample("W-SUN")
        weekly, counts = g.sum(), g.count()
        for week_end, total in weekly[counts == 7].items():
            first.setdefault(week_end, total)            # keep the earliest version only
    return pd.Series(first, name="first_release").sort_index()


def settled_outcomes(kalshi_path="kalshi_settled.csv", pw_dir="portwatch"):
    """Weekly settlement totals: Kalshi's own settlement value where available (the truth
    for the market), otherwise the first-release PortWatch total from the archive."""
    out = first_release_weekly(pw_dir) if os.path.isdir(pw_dir) else pd.Series(dtype=float)
    if os.path.exists(kalshi_path):
        k = pd.read_csv(kalshi_path, parse_dates=["week_end"])
        k["expiration_value"] = pd.to_numeric(k["expiration_value"], errors="coerce")
        kv = k.dropna(subset=["expiration_value"]).groupby("week_end")["expiration_value"].first()
        out = kv.combine_first(out)
    return out.sort_index().rename("outcome_total")


def score_forecasts(log_path="forecast_log.csv", pw_dir="portwatch", outcomes=None,
                    kalshi_path="kalshi_settled.csv", one_per_day=True,
                    save_path="scored_forecasts.csv", verbose=True):
    """Score logged forecasts against settled outcomes and save every scored row.

    one_per_day: keep only the last forecast of each UTC day per week/strike, so extra
    manual runs don't count more than the daily scheduled one."""
    if not os.path.exists(log_path):
        if verbose:
            print("No forecast log yet.")
        return pd.DataFrame()
    log = pd.read_csv(log_path, parse_dates=["logged_at", "week_end", "as_of"])
    if one_per_day:
        log["day"] = log["logged_at"].dt.normalize()
        log = (log.sort_values("logged_at")
                  .drop_duplicates(["day", "week_end", "strike"], keep="last")
                  .drop(columns="day"))
    # Only forecasts made before PortWatch published the week (Tuesday ~9 AM ET = 13:00 UTC)
    # count; later ones could already contain the answer through the ratio estimate.
    release = log["week_end"] + pd.Timedelta(days=2, hours=13)
    log = log[log["logged_at"] < release]
    outcomes = settled_outcomes(kalshi_path, pw_dir) if outcomes is None else outcomes
    log["outcome_total"] = log["week_end"].map(outcomes)
    scored = log.dropna(subset=["outcome_total"]).copy()
    if scored.empty:
        if verbose:
            print("No settled weeks in the log yet.")
        return scored
    scored["hit"] = (scored["outcome_total"] > scored["strike"]).astype(int)
    scored["brier_model"] = (scored["prob"] - scored["hit"]) ** 2
    scored["brier_market"] = (scored["market_prob"] - scored["hit"]) ** 2   # NaN where no price
    if "prob_ewma" in scored:
        scored["brier_ewma"] = (scored["prob_ewma"] - scored["hit"]) ** 2  # NaN for older rows
    if save_path:
        scored.to_csv(save_path, index=False)

    if verbose:
        both = scored.dropna(subset=["market_prob"])
        by_week = scored.groupby("week_end").agg(outcome=("outcome_total", "first"),
                                                 forecasts=("prob", "size"),
                                                 brier_model=("brier_model", "mean"))
        if not both.empty:
            by_week = by_week.join(both.groupby("week_end")[["brier_model", "brier_market"]]
                                   .mean().add_suffix("_priced"))
        print("Scores by settled week (lower is better):")
        print(by_week.round(4).to_string())
        if not both.empty:
            m, k = both["brier_model"].mean(), both["brier_market"].mean()
            print(f"Head-to-head where a price was recorded: model {m:.4f} vs market {k:.4f} "
                  f"({'model' if m < k else 'market'} ahead, {len(both)} forecasts, "
                  f"{both['week_end'].nunique()} weeks)")
        if "brier_ewma" in scored and scored["brier_ewma"].notna().any():
            e3 = scored.dropna(subset=["brier_ewma"])
            print(f"Model vs EWMA benchmark: model {e3['brier_model'].mean():.4f} vs "
                  f"EWMA {e3['brier_ewma'].mean():.4f} ({len(e3)} forecasts)")
    return scored


def maybe_update_estimations(path=ESTIMATIONS_PATH, every_days=7, **kwargs):
    """Run update_estimations() if it hasn't run automatically in the last every_days days."""
    with open(path, encoding="utf-8") as fh:
        est = json.load(fh)
    last = est.get("_last_auto_update")
    today = _dt.date.today()
    if last and (today - _dt.date.fromisoformat(last)).days < every_days:
        print(f"Estimations: last auto-update {last}, next due in "
              f"{every_days - (today - _dt.date.fromisoformat(last)).days} day(s)")
        return None
    print("Estimations: running weekly update")
    est = update_estimations(path=path, write=True, **kwargs)
    est["_last_auto_update"] = today.isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(est, fh, indent=2)
    return est

# Apply estimations.json at import time
load_estimations()
