# Panama Canal weekly transit forecast

A daily pipeline that forecasts weekly Panama Canal transit counts as reported by IMF PortWatch, prices the outcome as probabilities for each strike, and scores those forecasts against a prediction market and a naive benchmark.

![Dashboard](docs/dashboard.png)
<!-- Screenshots show a settled week from the private, tuned version. See "Parameters". -->

## The problem

The prediction market settles its weekly Panama Canal market on PortWatch's satellite-based (AIS) transit count. The Panama Canal Authority (ACP) publishes its own numbers every day: actual transits, schedules for the next few days, and booked slots weeks ahead. The two sources don't match. PortWatch counts fewer ships than the ACP, and the gap changes with the season and over time.

So the forecast is split into two parts:

1. **How many ships will the canal pass this week?** Estimated from ACP data, which is close to certain for past days and gets wider further out.
2. **What share of them will PortWatch count?** Estimated as a PortWatch/ACP ratio, tracked over time with an EWMA around a seasonal baseline.

Forecasting the ratio separately gives a much tighter forecast than modeling PortWatch's series on its own, because the ACP side carries real information about the coming week.

## How it works

```
ACP daily reports ─┐
ACP booking feeds ─┼─► collect_daily.py ─► forecast.py ─► run_forecast.py ─► dashboard.py
PortWatch (ArcGIS) ┤                        │                 │
The prediction market prices ─────┘                        │                 └─ compare with market prices, log
                                            ├─ daily ACP estimate per day of the week
                                            ├─ PortWatch / ACP ratio estimate
                                            ├─ Monte Carlo of the weekly PortWatch total
                                            └─ EWMA benchmark
```

**Daily ACP estimate.** Each day of the market week gets the best estimate available at forecast time:
- *Past days:* actual transits from the daily report
- *Next 0–2 days:* the published schedule, with error measured by horizon
- *About 3–13 days out:* booked slots plus an average gap for unbooked transits
- *Further out:* the ACP's daily slot cap plus an adjustment

**Ratio.** Weekly PortWatch/ACP ratios are compared with a monthly seasonal normal. The deviation is smoothed with an EWMA, and its uncertainty has a floor so the model never claims more precision than the data supports. Weeks with known data problems are excluded.

**Simulation.** 100,000 simulated weeks combine daily uncertainty, ratio uncertainty, and rare disruption days into a distribution of the PortWatch weekly total. That distribution gives the probability for each strike.

**Regime check.** The model assumes the canal is running at capacity, with demand exceeding available slots. If slots are going unbooked, it falls back to the EWMA benchmark.

**Scoring.** Each forecast is logged next to the market's mid price. After settlement, which uses PortWatch's *first* release, the model, the market, and the EWMA benchmark are scored with the Brier score.

## Files

| File | Purpose |
|---|---|
| `collect_daily.py` | Downloads ACP daily reports (PDF) and booking feeds, PortWatch data, and The prediction market prices. Keeps every published version of PortWatch data. |
| `forecast.py` | Loaders, daily estimates, ratio model, simulation, EWMA benchmark, logging, and scoring |
| `kalshi.py` | The prediction market public market data (no API key needed) |
| `run_forecast.py` | Daily run: forecast open weeks, compare with market prices, write the log and report |
| `dashboard.py` | Builds `reports/dashboard.html`: ratio control chart, week tracker, forecast distribution, model vs market, scorecard |
| `run_all.bat` | Runs collection and forecast. Can be scheduled with Windows Task Scheduler. |
| `estimations.example.json` | Template for the uncertainty parameters |

## Setup

```bash
pip install requests pypdf pandas numpy
cp estimations.example.json estimations.json
python collect_daily.py
python run_forecast.py
python dashboard.py
```

On Windows, schedule `run_all.bat` to run once a day. Set the `PYTHON` environment variable if `python` isn't on your PATH. The dashboard loads Plotly from a CDN, so it needs an internet connection to draw the charts.

Data source URLs in `collect_daily.py` (ACP daily reports, ACP booking feeds, PortWatch) are placeholders. All three are public; fill them in before running.

The model needs a few weeks of collected data before the ratio estimate and the measured uncertainties are meaningful.

## Parameters

The values in `SETTINGS` (`forecast.py`) and `estimations.example.json` are generic placeholders. The tuned values I use for live forecasts are kept private, so this repo runs end to end but won't reproduce my exact forecasts or scorecard.

To calibrate your own:
- `schedule_sd`, `gap_mean`, `gap_sd`: measure from ACP daily reports vs. actual transits
- `ratio_seasonal`: monthly PortWatch/ACP ratio from historical summaries
- `disruption_*`: estimate from observed disruption days

`update_estimations()` re-measures the parameters that the collected data supports.

## Limitations and next steps

- The track record is short. Brier score comparisons need many settled weeks before they mean much.
- Several parameters, such as the booking gap and disruption effects, rest on only a few observations.
- PortWatch revises its data after first release. The model scores against the first release because that's what settles the market.
- Next: model demand for the regime where the canal isn't at capacity, and widen the scorecard as more weeks settle.

*Data sources: Panama Canal Authority (public daily reports and booking feeds), IMF PortWatch, The prediction market public API. Not affiliated with any of them. Not financial advice.*
