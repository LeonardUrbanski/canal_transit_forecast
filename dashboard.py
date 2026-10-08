"""
dashboard.py - builds reports/dashboard.html (one page, opens in any browser).

Panels
  1. Ratio control chart: share of ACP transits counted by PortWatch, with the
     seasonal normal, +/-2 std band, monthly history, weekly ratios from the daily
     reports, the current estimate, and PortWatch revision markers.
  2. Week tracker (per open market week): daily estimates by source, and the
     running total against what each strike requires in ACP transits.
  3. Forecast distribution (per open week): simulated PortWatch totals with strike
     lines labelled with model and market probabilities.
  4. Model vs market over time (per open week), from forecast_log.csv.
  6. Scorecard: final forecast vs settled total for each week, and cumulative
     Brier scores of model vs market (from scored_forecasts.csv).

Charts use Plotly.js loaded from its CDN, so no extra Python packages are needed
(the page needs an internet connection to draw the charts).
Optional: acp_monthly.csv (columns: month, acp_daily_avg) adds the 2019+ monthly history.
"""

import glob
import html
import json
import os

import numpy as np
import pandas as pd

import forecast as f

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MONTHLY_SD = 0.013          # month-to-month std of the ratio around its seasonal normal (2019-2025)
SOURCE_STYLE = {            # bar colour by data source
    "actual": ("#1f4e79", "Actual (daily report)"),
    "schedule": ("#4a90d9", "Scheduled (daily report)"),
    "bookings": ("#a9c8ea", "Booked + gap"),
    "cap": ("#dde6f0", "Slot cap"),
}
STRIKE_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]


def _d(x):
    """Dates -> ISO strings for JSON."""
    return [pd.Timestamp(v).strftime("%Y-%m-%d %H:%M") for v in x]


def _num(x):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in x]


def _plot(div_id, traces, layout, height=380):
    layout = {"height": height, "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
              "legend": {"orientation": "h", "y": -0.2}, "hovermode": "x unified", **layout}
    # Charts with a range slider: slim the slider and drop the legend below it so they don't overlap.
    slider = layout.get("xaxis", {}).get("rangeslider")
    if slider and slider.get("visible"):
        slider.setdefault("thickness", 0.08)
        layout["legend"] = {"orientation": "h", "x": 0, "y": -0.32, "yanchor": "top"}
        layout["height"] = height + 40
    return (f'<div id="{div_id}"></div>\n<script>Plotly.newPlot("{div_id}", '
            f'{json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});</script>')


# ---------------------------------------------------------------- panel 1
def _revision_dates(pw_dir="portwatch"):
    files = sorted(glob.glob(os.path.join(pw_dir, "panama_*_thru_*.csv")))
    dates = []
    for a, b in zip(files, files[1:]):
        old = f.load_portwatch(a, glitch=None)
        new = f.load_portwatch(b, glitch=None)
        diff = (new - old).dropna()
        if (diff != 0).any():
            stamp = os.path.basename(b).split("_")[1]
            dates.append((stamp, int((diff != 0).sum())))
    return dates


def ratio_panel(results, pw, actuals, schedules, monthly_path="acp_monthly.csv"):
    traces, shapes, annotations = [], [], []
    x_end = max(r["week_end"] for r in results) + pd.Timedelta(days=14)

    # Monthly history (optional file)
    start = pd.Timestamp("2025-07-01")
    if os.path.exists(monthly_path):
        m = pd.read_csv(monthly_path, parse_dates=["month"]).set_index("month")
        pw_m = pw.resample("MS").mean()
        ratio_m = (pw_m / m["acp_daily_avg"]).dropna()
        mid = ratio_m.index + pd.Timedelta(days=14)
        traces.append({"x": _d(mid), "y": _num(ratio_m.values), "name": "Monthly ratio (ACP summaries)",
                       "mode": "lines+markers", "line": {"color": "#555"}, "marker": {"size": 5}})
        start = ratio_m.index.min()

    # Seasonal normal and +/-2 std band (daily resolution so month steps line up)
    days = pd.date_range(start, x_end, freq="D")
    normal = np.array([f.SETTINGS["ratio_seasonal"][d.month] for d in days])
    traces.append({"x": _d(days), "y": _num(normal + 2 * MONTHLY_SD), "mode": "lines",
                   "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"})
    traces.append({"x": _d(days), "y": _num(normal - 2 * MONTHLY_SD), "mode": "lines",
                   "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(255,165,0,0.15)",
                   "name": "Normal range (±2 std, monthly)", "hoverinfo": "skip"})
    traces.append({"x": _d(days), "y": _num(normal), "mode": "lines", "name": "Typical for the month",
                   "line": {"color": "orange", "dash": "dash"}})

    # Weekly ratios from daily reports (excluded weeks shown hollow)
    weeks = f.weekly_ratio_table(pw, actuals, schedules)
    if not weeks.empty:
        excl = f.SETTINGS.get("ratio_exclude_weeks", {})
        used = weeks[~weeks.index.strftime("%Y-%m-%d").isin(excl)]
        skipped = weeks[weeks.index.strftime("%Y-%m-%d").isin(excl)]
        x = used.index - pd.Timedelta(days=3)           # plot at the week's Thursday
        traces.append({"x": _d(x), "y": _num(used["ratio"]), "name": "Weekly ratio (daily reports)",
                       "mode": "lines+markers", "line": {"color": "#1f77b4"}, "marker": {"size": 9},
                       "text": [f"week ending {w.date()}: {int(p)} PW / {int(a)} ACP"
                                for w, p, a in zip(used.index, used["portwatch"], used["acp"])],
                       "hovertemplate": "%{text}<br>ratio %{y:.3f}<extra></extra>"})
        if not skipped.empty:
            traces.append({"x": _d(skipped.index - pd.Timedelta(days=3)), "y": _num(skipped["ratio"]),
                           "name": "Excluded week", "mode": "markers",
                           "marker": {"size": 9, "symbol": "circle-open", "color": "#999"},
                           "text": [excl[w.strftime("%Y-%m-%d")] for w in skipped.index],
                           "hovertemplate": "%{text}<br>ratio %{y:.3f}<extra></extra>"})

    # Current estimate for each forecast week
    for r in results:
        x = r["week_end"] - pd.Timedelta(days=3)
        traces.append({"x": _d([x]), "y": [r["ratio"]["ratio"]], "mode": "markers",
                       "name": f"Estimate, week ending {r['week_end'].date()}",
                       "marker": {"size": 11, "symbol": "diamond", "color": "#d62728"},
                       "error_y": {"type": "data", "array": [2 * r["ratio"]["sd"]], "visible": True}})

    # PortWatch revisions
    for stamp, n in _revision_dates():
        shapes.append({"type": "line", "x0": stamp, "x1": stamp, "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": "purple", "dash": "dot", "width": 1}})
        annotations.append({"x": stamp, "yref": "paper", "y": 1, "text": f"PW revision ({n} days)",
                            "showarrow": False, "font": {"size": 10, "color": "purple"}})

    view_start = max(start, pd.Timestamp(x_end) - pd.Timedelta(days=455))
    layout = {"title": "Share of ACP transits counted by PortWatch",
              "yaxis": {"title": "PortWatch / ACP", "tickformat": ".2f"},
              "xaxis": {"range": _d([view_start, x_end]), "rangeslider": {"visible": True}},
              "shapes": shapes, "annotations": annotations, "hovermode": "closest"}
    return _plot("ratio", traces, layout, height=480)


# ---------------------------------------------------------------- panel 2
def week_panels(r, idx):
    daily = r["daily"].copy()
    daily["kind"] = daily["source"].str.replace(r"_h\d", "", regex=True)
    ratio, ratio_sd = r["ratio"]["ratio"], r["ratio"]["sd"]
    daily["pw"] = daily["mean"] * ratio                       # expected PortWatch count
    daily["missed"] = daily["mean"] - daily["pw"]             # ACP transits PortWatch won't count
    daily["pw_sd"] = np.sqrt((ratio * daily["sd"]) ** 2 + (daily["mean"] * ratio_sd) ** 2)

    # --- daily stacked bars: PortWatch part (coloured by source) + uncounted part
    traces = []
    for kind, (color, label) in SOURCE_STYLE.items():
        part = daily[daily["kind"] == kind]
        if part.empty:
            continue
        traces.append({"type": "bar", "x": _d(part.index), "y": _num(part["pw"]),
                       "name": f"{label}: counted by PortWatch",
                       "marker": {"color": color, "line": {"color": "#1f4e79", "width": 1}},
                       "customdata": [[m, sd, p, psd, d] for m, sd, p, psd, d in
                                      zip(part["mean"], part["sd"], part["pw"], part["pw_sd"], part["detail"])],
                       "hovertemplate": ("ACP %{customdata[0]:.0f} ± %{customdata[1]:.1f}<br>"
                                         "PortWatch ≈ %{customdata[2]:.1f} ± %{customdata[3]:.1f}<br>"
                                         "%{customdata[4]}<extra></extra>")})
    traces.append({"type": "bar", "x": _d(daily.index), "y": _num(daily["missed"]),
                   "name": f"Not counted by PortWatch (1 − ratio {ratio:.3f})",
                   "marker": {"color": "rgba(200,200,200,0.6)", "pattern": {"shape": "/"},
                              "line": {"color": "#888", "width": 1}},
                   "error_y": {"type": "data", "array": _num(daily["sd"]), "visible": True,
                               "color": "#222", "thickness": 1.5, "width": 6},
                   "hovertemplate": "not counted ≈ %{y:.1f}<br>error bar: ACP ±1 std<extra></extra>"})
    bars = _plot(f"days{idx}", traces,
                 {"title": "Daily transits: ACP total (full bar, ±1 std) and the part PortWatch counts",
                  "yaxis": {"title": "transits", "rangemode": "tozero"}, "barmode": "stack",
                  "hovermode": "closest"})

    # --- running totals: ACP and PortWatch, each with a ±2 std band
    cum = daily["mean"].cumsum()
    cum_sd = np.sqrt((daily["sd"] ** 2).cumsum())
    pw_cum = cum * ratio
    pw_cum_sd = np.sqrt((ratio * cum_sd) ** 2 + (cum * ratio_sd) ** 2)   # canal + ratio uncertainty
    x = _d(daily.index)
    traces = [
        {"x": x, "y": _num(cum + 2 * cum_sd), "mode": "lines", "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
        {"x": x, "y": _num(cum - 2 * cum_sd), "mode": "lines", "line": {"width": 0}, "fill": "tonexty",
         "fillcolor": "rgba(31,119,180,0.12)", "name": "ACP ±2 std", "hoverinfo": "skip"},
        {"x": x, "y": _num(cum), "mode": "lines+markers", "name": "ACP running total",
         "line": {"color": "#1f77b4"}},
        {"x": x, "y": _num(pw_cum + 2 * pw_cum_sd), "mode": "lines", "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
        {"x": x, "y": _num(pw_cum - 2 * pw_cum_sd), "mode": "lines", "line": {"width": 0}, "fill": "tonexty",
         "fillcolor": "rgba(214,39,40,0.12)", "name": "PortWatch ±2 std (incl. ratio)", "hoverinfo": "skip"},
        {"x": x, "y": _num(pw_cum), "mode": "lines+markers", "name": "PortWatch running total (expected)",
         "line": {"color": "#d62728"},
         "customdata": _num(pw_cum_sd),
         "hovertemplate": "%{y:.0f} ± %{customdata:.1f}<extra>PortWatch</extra>"},
    ]
    shapes, annotations = [], []
    final_pw = float(pw_cum.iloc[-1])
    for i, k in enumerate(r["sim"]["probs"]):
        if abs(k - final_pw) > 40:
            continue
        c = STRIKE_COLORS[i % len(STRIKE_COLORS)]
        y = k + 0.5
        shapes.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": y, "y1": y,
                       "line": {"color": c, "dash": "dash"}})
        annotations.append({"xref": "paper", "x": 0.01, "y": y, "yanchor": "bottom", "showarrow": False,
                            "text": f"PortWatch above {k:g} (≈{y / ratio:.0f} ACP) — model {r['sim']['probs'][k]:.0%}",
                            "font": {"color": c, "size": 11}})
    total = _plot(f"cum{idx}", traces,
                  {"title": f"Running totals vs. strikes (ratio {ratio:.3f} ± {ratio_sd:.3f})",
                   "yaxis": {"title": "cumulative transits"},
                   "shapes": shapes, "annotations": annotations})
    return bars + total


# ---------------------------------------------------------------- panel 3
def distribution_panel(r, prices, idx):
    """Probability of each possible PortWatch total, with strike lines."""
    sims = np.asarray(r["sim"]["sims"])
    values, counts = np.unique(sims, return_counts=True)
    probs = counts / counts.sum()
    keep = probs >= 0.0005                                  # drop the far tails for readability
    values, probs = values[keep], probs[keep]

    w = None
    if prices is not None and not prices.empty:
        w = prices[prices["week_end"] == r["week_end"]].set_index("strike")

    traces = [{"type": "bar", "x": _num(values), "y": _num(probs), "name": "Simulated PortWatch total",
               "marker": {"color": "#7fa7d6"},
               "hovertemplate": "total %{x:.0f}: %{y:.1%}<extra></extra>"}]
    shapes, annotations = [], []
    lo, hi = float(values.min()), float(values.max())
    for i, (k, p) in enumerate(r["sim"]["probs"].items()):
        if not (lo - 10 <= k <= hi + 10):
            continue
        c = STRIKE_COLORS[i % len(STRIKE_COLORS)]
        x = k + 0.5                                         # boundary: "above k" means k+1 or more
        shapes.append({"type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"color": c, "dash": "dash", "width": 2}})
        mkt = ""
        if w is not None and k in w.index and pd.notna(w.loc[k, "mid"]):
            mkt = f"<br>market {w.loc[k, 'mid']:.0%}"
        annotations.append({"x": x, "yref": "paper", "y": 1.0, "xanchor": "left", "yanchor": "top",
                            "showarrow": False, "align": "left",
                            "text": f"<b>above {k:g}</b><br>model {p:.1%}{mkt}",
                            "font": {"color": c, "size": 11}, "bgcolor": "rgba(255,255,255,0.8)"})
    mean = r["sim"]["mean"]
    shapes.append({"type": "line", "x0": mean, "x1": mean, "yref": "paper", "y0": 0, "y1": 0.85,
                   "line": {"color": "#333", "width": 1}})
    layout = {"title": f"Forecast distribution of the PortWatch total (mean {mean:.0f})",
              "xaxis": {"title": "PortWatch weekly total", "range": [min(lo, mean - 30), max(hi, mean + 30)]},
              "yaxis": {"title": "probability", "tickformat": ".0%"},
              "shapes": shapes, "annotations": annotations, "hovermode": "closest", "bargap": 0.05,
              "showlegend": False}
    return _plot(f"dist{idx}", traces, layout, height=360)


# ---------------------------------------------------------------- panel 4
def market_panel(r, idx, log_path="forecast_log.csv"):
    if not os.path.exists(log_path):
        return "<p>No forecast log yet.</p>"
    log = pd.read_csv(log_path, parse_dates=["logged_at", "week_end"])
    log = log[log["week_end"] == r["week_end"]].sort_values("logged_at")
    if log.empty:
        return "<p>No logged forecasts for this week yet.</p>"
    traces = []
    for i, (k, g) in enumerate(log.groupby("strike")):
        c = STRIKE_COLORS[i % len(STRIKE_COLORS)]
        traces.append({"x": _d(g["logged_at"]), "y": _num(g["prob"]), "mode": "lines+markers",
                       "name": f"{k:g} model", "line": {"color": c}})
        if g["market_prob"].notna().any():
            traces.append({"x": _d(g["logged_at"]), "y": _num(g["market_prob"]), "mode": "lines+markers",
                           "name": f"{k:g} market", "line": {"color": c, "dash": "dash"},
                           "marker": {"symbol": "x"}})
    return _plot(f"mkt{idx}", traces, {"title": "Model (solid) vs. market mid (dashed) by strike",
                                        "yaxis": {"title": "P(above strike)", "range": [0, 1], "tickformat": ".0%"}})


# ---------------------------------------------------------------- panel 6
def scorecard_panel(actuals, schedules, scored_path="scored_forecasts.csv"):
    """Predicted vs actual for each settled week, and model vs market Brier scores."""
    if not os.path.exists(scored_path):
        return "<p>No settled weeks yet.</p>"
    sc = pd.read_csv(scored_path, parse_dates=["logged_at", "week_end"])
    if sc.empty:
        return "<p>No settled weeks yet.</p>"

    # Final pre-release forecast for each week (one row per week)
    final = (sc.sort_values("logged_at").groupby("week_end").tail(1).set_index("week_end").sort_index())
    acp = f.acp_daily_series(actuals, schedules)["acp"]
    acp_week = acp.resample("W-SUN").sum()
    rows, weeks = [], []
    for w, r in final.iterrows():
        g = sc[sc["week_end"] == w]
        priced = g.dropna(subset=["market_prob"])
        last_day = g[g["logged_at"] == g["logged_at"].max()]
        lp = last_day.dropna(subset=["market_prob"])
        lo, hi = r["pw_mean"] - 1.645 * r["pw_sd"], r["pw_mean"] + 1.645 * r["pw_sd"]
        actual_ratio = r["outcome_total"] / acp_week.get(w, np.nan) if w in acp_week.index else np.nan
        e3 = g.dropna(subset=["brier_ewma"]) if "brier_ewma" in g else g.iloc[0:0]
        weeks.append({"week": w, "pred": r["pw_mean"], "sd": r["pw_sd"], "actual": r["outcome_total"],
                      "ewma_pred": r.get("ewma_mean", np.nan),
                      "brier_ewma": e3["brier_ewma"].mean() if not e3.empty else np.nan,
                      "brier_model_e": e3["brier_model"].mean() if not e3.empty else np.nan,
                      "brier_model": g["brier_model"].mean(),
                      "brier_model_p": priced["brier_model"].mean() if not priced.empty else np.nan,
                      "brier_market_p": priced["brier_market"].mean() if not priced.empty else np.nan})
        inside = "✓" if lo <= r["outcome_total"] <= hi else "✗"
        def fmt(v, spec):
            return "–" if pd.isna(v) else format(v, spec)
        rows.append(
            f"<tr><td>{w:%b %d}</td><td>{r['logged_at']:%b %d %H:%M}</td>"
            f"<td>{r['pw_mean']:.0f} ± {r['pw_sd']:.1f}</td><td>{lo:.0f}–{hi:.0f}</td>"
            f"<td><b>{r['outcome_total']:.0f}</b></td><td>{r['outcome_total'] - r['pw_mean']:+.0f}</td>"
            f"<td>{inside}</td><td>{r['ratio']:.3f}</td><td>{fmt(actual_ratio, '.3f')}</td>"
            f"<td>{fmt(lp['brier_model'].mean() if not lp.empty else np.nan, '.4f')}</td>"
            f"<td>{fmt(lp['brier_market'].mean() if not lp.empty else np.nan, '.4f')}</td>"
            f"<td>{len(g['logged_at'].unique())}</td></tr>")
    table = ("<table><tr><th>Week ending</th><th>Final forecast at</th><th>Predicted</th><th>90% range</th>"
             "<th>Actual</th><th>Error</th><th>In range</th><th>Ratio (pred)</th><th>Ratio (actual)</th>"
             "<th>Brier model*</th><th>Brier market*</th><th>Forecast days</th></tr>"
             + "".join(rows) + "</table><p class='muted'>* final forecast, strikes with a recorded price. "
             "Actual ratio = settled PortWatch total ÷ ACP total from the daily reports.</p>")

    wk = pd.DataFrame(weeks).set_index("week")
    x = _d(wk.index)
    pred = _plot("score_pred", [
        {"x": x, "y": _num(wk["pred"]), "mode": "markers", "name": "Predicted (±2 std)",
         "marker": {"size": 9, "color": "#1f77b4"},
         "error_y": {"type": "data", "array": _num(2 * wk["sd"]), "visible": True}},
        {"x": x, "y": _num(wk["ewma_pred"]), "mode": "markers", "name": "EWMA benchmark",
         "marker": {"size": 8, "symbol": "square-open", "color": "#7f7f7f"}},
        {"x": x, "y": _num(wk["actual"]), "mode": "markers", "name": "Actual (settled)",
         "marker": {"size": 11, "symbol": "x", "color": "#d62728"}},
    ], {"title": "Final forecast vs. settled PortWatch total", "yaxis": {"title": "weekly total"},
        "hovermode": "x unified"}, height=330)

    cum_model = wk["brier_model_p"].expanding().mean()
    cum_market = wk["brier_market_p"].expanding().mean()
    brier = _plot("score_brier", [
        {"x": x, "y": _num(cum_model), "mode": "lines+markers", "name": "Model (cumulative)", "line": {"color": "#1f77b4"}},
        {"x": x, "y": _num(cum_market), "mode": "lines+markers", "name": "Market (cumulative)",
         "line": {"color": "#ff7f0e", "dash": "dash"}},
        {"x": x, "y": _num(wk["brier_ewma"].expanding().mean()), "mode": "lines+markers",
         "name": "EWMA benchmark (cumulative, all strikes)", "line": {"color": "#7f7f7f", "dash": "dot"}},
    ], {"title": "Brier score, all daily forecasts with a market price (lower is better)",
        "yaxis": {"title": "mean Brier score", "rangemode": "tozero"}}, height=330)
    n = len(wk)
    note = (f"<p class='muted'>{n} settled week{'s' if n != 1 else ''}. "
            "Early results are mostly noise; look for a consistent gap over many weeks.</p>")
    return table + note + pred + brier


# ---------------------------------------------------------------- traffic tab
RANGE_BUTTONS = {"buttons": [
    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
    {"count": 2, "label": "2Y", "step": "year", "stepmode": "backward"},
    {"step": "all", "label": "All"}]}
TYPE_COLORS = {"container": "#1f77b4", "dry_bulk": "#8c564b", "general_cargo": "#9467bd",
               "roll-on/roll-off": "#2ca02c", "tanker": "#ff7f0e"}


def _time_layout(title, ytitle, start_months=12):
    return {"title": title, "yaxis": {"title": ytitle},
            "xaxis": {"rangeselector": RANGE_BUTTONS, "rangeslider": {"visible": True},
                      "range": _d([pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=start_months),
                                   pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=7)])}}


def traffic_panels(actuals, schedules, pw, pw_types_path="portwatch_panama_latest.csv",
                   monthly_path="acp_monthly.csv"):
    out = []
    # PortWatch weekly by vessel type
    raw = pd.read_csv(pw_types_path)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    types = [c for c in TYPE_COLORS if c in raw.columns]
    daily_t = raw.set_index("datetime")[types].astype(float)
    daily_t.loc["2020-01-26":"2020-02-02"] = np.nan              # known glitch
    g = daily_t.resample("W-SUN")
    weekly_t = g.sum()[g.count().min(axis=1) == 7]
    traces = [{"x": _d(weekly_t.index), "y": _num(weekly_t[t]), "name": t.replace("_", " "),
               "stackgroup": "pw", "mode": "lines", "line": {"width": 0.5, "color": TYPE_COLORS[t]}}
              for t in types]
    out.append(_plot("tr_types", traces, _time_layout("PortWatch weekly transits by vessel type (stacked)",
                                                       "transits per week"), height=430))

    # Canal (ACP) vs PortWatch, transits per day
    traces = []
    pw_m = pw.resample("MS").mean()
    traces.append({"x": _d(pw_m.index + pd.Timedelta(days=14)), "y": _num(pw_m), "name": "PortWatch, monthly avg/day",
                   "mode": "lines+markers", "line": {"color": "#d62728"}, "marker": {"size": 4}})
    if os.path.exists(monthly_path):
        m = pd.read_csv(monthly_path, parse_dates=["month"]).set_index("month")["acp_daily_avg"]
        traces.append({"x": _d(m.index + pd.Timedelta(days=14)), "y": _num(m), "name": "ACP, monthly avg/day (summaries)",
                       "mode": "lines+markers", "line": {"color": "#1f77b4"}, "marker": {"size": 4}})
    acp_w = f.acp_daily_series(actuals, schedules)["acp"]
    g = acp_w.resample("W-SUN")
    acp_week = (g.sum() / 7)[g.count() == 7]
    pw_week = pw_weekly(pw) / 7
    traces.append({"x": _d(acp_week.index - pd.Timedelta(days=3)), "y": _num(acp_week),
                   "name": "ACP, weekly avg/day (daily reports)", "mode": "markers",
                   "marker": {"size": 8, "color": "#1f77b4", "symbol": "diamond"}})
    traces.append({"x": _d(pw_week.index - pd.Timedelta(days=3)), "y": _num(pw_week),
                   "name": "PortWatch, weekly avg/day", "mode": "markers",
                   "marker": {"size": 6, "color": "#d62728", "symbol": "circle-open"}})
    out.append(_plot("tr_canal", traces, _time_layout("Canal (ACP) vs PortWatch: transits per day",
                                                       "transits per day", start_months=24), height=430))
    return "".join(out)


def pw_weekly(pw_daily):
    g = pw_daily.resample("W-SUN")
    return g.sum()[g.count() == 7]


# ---------------------------------------------------------------- page
def _summary_table(r, prices):
    rows = []
    w = prices[prices["week_end"] == r["week_end"]].set_index("strike") if prices is not None and not prices.empty else None
    ewma = r.get("ewma")
    for k, p in r["sim"]["probs"].items():
        bid = ask = mid = None
        if w is not None and k in w.index:
            bid, ask, mid = w.loc[k, "yes_bid"], w.loc[k, "yes_ask"], w.loc[k, "mid"]
        fmt = lambda v: "–" if v is None or pd.isna(v) else f"{v:.0%}"
        diff = "" if mid is None or pd.isna(mid) else f"{(p - mid) * 100:+.0f} pts"
        e = f"{ewma['probs'][k]:.1%}" if ewma else "–"
        rows.append(f"<tr><td>{k:g}</td><td><b>{p:.1%}</b></td><td>{e}</td><td>{fmt(bid)}</td><td>{fmt(ask)}</td>"
                    f"<td>{fmt(mid)}</td><td>{diff}</td></tr>")
    sim = r["sim"]
    regime = r["regime"]["regime"]
    regime_txt = "regime 1 (sold out)" if regime == 1 else f"<span class='warn'>regime {regime}</span>"
    return (f"<p>Expected PortWatch total <b>{sim['mean']:.0f}</b> ± {sim['sd']:.1f} "
            f"(90% range {sim['quantiles'][5]:.0f}–{sim['quantiles'][95]:.0f}); "
            f"canal {r['canal_mean']:.0f} ± {r['canal_sd']:.1f}; ratio {r['ratio']['ratio']:.3f} ± {r['ratio']['sd']:.3f}; "
            f"{regime_txt}</p>"
            + (f"<p class='muted'>EWMA benchmark: {ewma['mean']:.0f} ± {ewma['sd']:.1f} "
               f"(from PortWatch history only)</p>" if ewma else "")
            + ("<p class='warn'>Fallback: regime is not 1, the EWMA is the primary forecast this week.</p>"
               if r.get("primary") == "ewma" else "")
            + "<table><tr><th>Strike</th><th>Model</th><th>EWMA</th><th>Bid</th><th>Ask</th><th>Mid</th><th>Model − mid</th></tr>"
            + "".join(rows) + "</table>")


def build_dashboard(results, prices=None, folder="reports/dashboards", alerts=(),
                    dcr_path="dcr_parsed.csv", pw_path="portwatch_panama_latest.csv",
                    latest_copy="reports/dashboard_latest.html"):
    """Writes a dated page (never overwritten) and, if latest_copy is set, a copy of the
    newest page under a fixed name for quick access. Returns the dated path."""
    actuals, schedules = f.load_dcr(dcr_path)
    pw = f.load_portwatch(pw_path)
    as_of = max(r["as_of"] for r in results)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"dashboard_{as_of:%Y-%m-%d_%H%M}.html")

    tabs = {}
    head = [f"<h1>Panama Canal weekly transits — dashboard</h1>",
            f"<p class='muted'>Generated {as_of:%Y-%m-%d %H:%M} UTC · model {f.SETTINGS['model_version']}</p>"]
    head += [f"<p class='warn'>⚠ {html.escape(a)}</p>" for a in alerts]

    ordered = sorted(results, key=lambda r: r["week_end"])
    overview = []
    for r in ordered:
        overview.append(f"<h2>Week ending {r['week_end']:%a %b %d}</h2>")
        overview.append(_summary_table(r, prices))
    tabs["Overview"] = "".join(overview)

    weeks_html = []
    for i, r in enumerate(ordered):
        weeks_html.append(f"<h2>Week ending {r['week_end']:%a %b %d}</h2>")
        weeks_html.append("<h3>Week tracker</h3>" + week_panels(r, i))
        weeks_html.append("<h3>Forecast distribution</h3>" + distribution_panel(r, prices, i))
        weeks_html.append("<h3>Model vs. market</h3>" + market_panel(r, i))
    tabs["Weeks"] = "".join(weeks_html)

    tabs["PortWatch ratio"] = ratio_panel(results, pw, actuals, schedules)
    try:
        tabs["Traffic"] = traffic_panels(actuals, schedules, pw)
    except Exception as e:
        tabs["Traffic"] = f"<p class='warn'>Traffic charts failed: {html.escape(str(e))}</p>"
    try:
        tabs["Scorecard"] = scorecard_panel(actuals, schedules)
    except Exception as e:
        tabs["Scorecard"] = f"<p class='warn'>Scorecard failed: {html.escape(str(e))}</p>"

    buttons = "".join(f'<button class="tab" onclick="showTab({i})">{html.escape(n)}</button>'
                      for i, n in enumerate(tabs))
    panes = "".join(f'<div class="pane" id="pane{i}">{body}</div>' for i, body in enumerate(tabs.values()))
    parts = head + [f'<div class="tabs">{buttons}</div>', panes, """
<script>
function showTab(i) {
  document.querySelectorAll('.pane').forEach((p, j) => p.style.display = (i === j) ? 'block' : 'none');
  document.querySelectorAll('.tab').forEach((b, j) => b.classList.toggle('active', i === j));
  document.querySelectorAll('#pane' + i + ' .js-plotly-plot').forEach(el => Plotly.Plots.resize(el));
}
showTab(0);
</script>"""]

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canal dashboard</title>
<script src="{PLOTLY_CDN}"></script>
<style>
 body {{ font-family: system-ui, Segoe UI, Arial, sans-serif; max-width: 1150px; margin: 20px auto; padding: 0 16px; color: #222; }}
 h1 {{ margin-bottom: 0; }} h2 {{ margin-top: 36px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
 .muted {{ color: #777; margin-top: 4px; }} .warn {{ color: #b00020; font-weight: 600; }}
 table {{ border-collapse: collapse; margin: 8px 0 16px; }}
 .tabs {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 2px solid #ddd; z-index: 10; }}
 .tab {{ font-size: 15px; padding: 8px 16px; margin-right: 4px; border: 1px solid #ccc; border-bottom: none;
        background: #f4f6f8; border-radius: 6px 6px 0 0; cursor: pointer; }}
 .tab.active {{ background: #1f4e79; color: #fff; border-color: #1f4e79; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 12px; text-align: right; }} th {{ background: #f4f6f8; }}
</style></head><body>
{''.join(parts)}
</body></html>"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    if latest_copy:
        os.makedirs(os.path.dirname(latest_copy) or ".", exist_ok=True)
        with open(latest_copy, "w", encoding="utf-8") as fh:
            fh.write(page)
    return out
