"""
Daily collector for Panama Canal data. Run once a day.

1. Daily Customer Reports: checks the archive (dcr-01 ... dcr-31) AND the
   current report (dcr.pdf), saves any report not seen before (named by the
   date printed inside), and appends parsed fields to dcr_parsed.csv.
2. Saves raw, timestamped snapshots of both booking feeds.
4. Kalshi (via kalshi.py): appends live prices for open weeks to kalshi_prices.csv and
   keeps kalshi_settled.csv current with results.
3. PortWatch: downloads the Panama Canal daily transit series from PortWatch's
   public ArcGIS service. Writes portwatch_panama_latest.csv (same columns as
   the manual download) and, whenever the data changed, a dated copy in
   portwatch/ so you keep every published version (first releases settle Kalshi).

Requires: pip install requests pypdf
"""

import csv, hashlib, io, os, re, time
from datetime import datetime, timedelta, timezone

import requests
from pypdf import PdfReader

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # always save next to this script

DCR_BASE = "<ACP daily customer report base URL>"   # folder holding dcr-01.pdf ... dcr-31.pdf and dcr.pdf
DCR_URLS = ([f"{DCR_BASE}/dcr-{d:02d}.pdf" for d in range(1, 32)]
            + [f"{DCR_BASE}/dcr.pdf"])
BOOKING_FEEDS = {
    "regular": "<ACP booking feed URL - regular>",
    "neo": "<ACP booking feed URL - neo>",
}
DCR_DIR, BOOKING_DIR, PARSED_CSV = "dcr", "booking_snapshots", "dcr_parsed.csv"
PW_URL = "<PortWatch daily chokepoints ArcGIS FeatureServer query URL>"
PW_WHERE = "portname LIKE '%Panama%'"
PW_DIR, PW_LATEST = "portwatch", "portwatch_panama_latest.csv"
PW_COLUMNS = {"n_container": "Container", "n_dry_bulk": "Dry Bulk",
              "n_general_cargo": "General Cargo", "n_roro": "Roll-on/roll-off",
              "n_tanker": "Tanker"}
HEADERS = {"User-Agent": "canal-traffic-research script"}
PAUSE = 1.5

FIELDS = ["report_date", "prepared", "actual_date", "actual_arrivals", "actual_transits",
          "sched_today", "sched_tomorrow", "sched_next",
          "due_today", "due_tomorrow", "due_next",
          "queue_0001", "queue_24h", "queue_48h", "source", "file"]


def pdf_text(data):
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)


def lone_ints(block):
    return [int(line.strip()) for line in block.splitlines() if re.fullmatch(r"\s*\d+\s*", line)]


def parse_dcr(text):
    m = re.search(r"REPORT FOR\s+(\d{2}-[A-Z]{3}-\d{4})", text)
    if not m:
        return None
    report = datetime.strptime(m.group(1), "%d-%b-%Y").date()
    out = {"report_date": report.isoformat(),
           "actual_date": (report - timedelta(days=1)).isoformat()}

    m = re.search(r"Prepared on:.*?(\d{2}-[A-Z]{3}-\d{4}\s+\d{4})", text, re.S)
    out["prepared"] = m.group(1) if m else None

    m = re.search(r"ACTUAL ARRIVALS\s+ACTUAL TRANSITS\s+(\d+)\s+(\d+)", text)
    out["actual_arrivals"], out["actual_transits"] = (int(m[1]), int(m[2])) if m else (None, None)

    due = text.split("A. VESSELS DUE")[0]
    sched = (text.split("A. VESSELS DUE")[-1].split("B. VESSELS SCHEDULED")[0]
             if "B. VESSELS SCHEDULED" in text else "")
    d, s = lone_ints(due)[-3:], lone_ints(sched)[-3:]
    out.update(zip(["due_today", "due_tomorrow", "due_next"], d if len(d) == 3 else [None] * 3))
    out.update(zip(["sched_today", "sched_tomorrow", "sched_next"], s if len(s) == 3 else [None] * 3))

    m = re.search(r"TOTALS\s+(\d+)\s+(\d+)\s+(\d+)\s*\n?\s*E\. QUEUE", text)
    out["queue_0001"], out["queue_24h"], out["queue_48h"] = (
        (int(m[1]), int(m[2]), int(m[3])) if m else (None, None, None))
    return out


def collect_dcr():
    os.makedirs(DCR_DIR, exist_ok=True)
    seen = set()
    if os.path.exists(PARSED_CSV):
        with open(PARSED_CSV, newline="") as f:
            seen = {row["report_date"] for row in csv.DictReader(f)}

    new_rows = []
    for url in DCR_URLS:
        name = url.rsplit("/", 1)[-1]
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  {name}: request failed ({e})")
            continue
        time.sleep(PAUSE)
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            continue

        row = parse_dcr(pdf_text(r.content))
        if row is None:
            print(f"  {name}: no report date found, saved as unparsed")
            with open(os.path.join(DCR_DIR, f"unparsed_{name}"), "wb") as f:
                f.write(r.content)
            continue
        if row["report_date"] in seen:
            continue

        path = os.path.join(DCR_DIR, f"dcr_{row['report_date']}.pdf")
        with open(path, "wb") as f:
            f.write(r.content)
        row.update(source=name, file=path)
        seen.add(row["report_date"])
        new_rows.append(row)
        missing = [k for k, v in row.items() if v is None]
        print(f"  {row['report_date']} ({name}): transits yesterday={row['actual_transits']}, "
              f"scheduled={row['sched_today']}/{row['sched_tomorrow']}/{row['sched_next']}"
              + (f"  (missing: {', '.join(missing)})" if missing else ""))

    if new_rows:
        write_header = not os.path.exists(PARSED_CSV)
        with open(PARSED_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for row in sorted(new_rows, key=lambda r: r["report_date"]):
                w.writerow(row)
    print(f"Daily reports: {len(new_rows)} new")


def collect_booking():
    os.makedirs(BOOKING_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    for name, url in BOOKING_FEEDS.items():
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Booking feed '{name}' failed: {e}")
            continue
        path = os.path.join(BOOKING_DIR, f"booking_{name}_{stamp}.json")
        with open(path, "wb") as f:
            f.write(r.content)  # raw bytes; parse later with .decode("utf-8-sig")
        print(f"Booking feed '{name}': saved {len(r.content):,} bytes to {path}")
        time.sleep(PAUSE)


def collect_portwatch():
    """Page through PortWatch's Panama rows (max 1,000 per request)."""
    rows, offset = [], 0
    while True:
        params = {"where": PW_WHERE, "outFields": "*", "orderByFields": "date ASC",
                  "resultOffset": offset, "resultRecordCount": 1000, "f": "json"}
        try:
            r = requests.get(PW_URL, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"PortWatch failed: {e}")
            return
        if "error" in data:
            print(f"PortWatch error: {data['error']}")
            return
        batch = [f["attributes"] for f in data.get("features", [])]
        rows.extend(batch)
        if not batch or not data.get("exceededTransferLimit"):
            break
        offset += len(batch)
        time.sleep(PAUSE)

    if not rows:
        print("PortWatch: no rows returned, check PW_WHERE")
        return
    names = sorted({r.get("portname") for r in rows})
    if len(names) != 1:
        print(f"PortWatch: query matched several chokepoints {names}; tighten PW_WHERE")
        return

    out_rows = []
    for r in rows:
        raw = r["date"]
        if isinstance(raw, (int, float)):
            day = datetime.fromtimestamp(raw / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            day = str(raw)[:10]          # text like "2026-09-27 00:00:00"
        row = {"DateTime": day}
        for src, dst in PW_COLUMNS.items():
            row[dst] = r.get(src)
        row["n_total"] = r.get("n_total")
        out_rows.append(row)
    out_rows.sort(key=lambda x: x["DateTime"])

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)
    content = buf.getvalue()

    old = open(PW_LATEST, encoding="utf-8").read() if os.path.exists(PW_LATEST) else None
    with open(PW_LATEST, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    last_day = out_rows[-1]["DateTime"]
    if old is None or hashlib.md5(old.encode()).hexdigest() != hashlib.md5(content.encode()).hexdigest():
        os.makedirs(PW_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
        path = os.path.join(PW_DIR, f"panama_{stamp}_thru_{last_day}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        print(f"PortWatch ({names[0]}): {len(out_rows)} days through {last_day}, NEW version saved to {path}")
    else:
        print(f"PortWatch ({names[0]}): {len(out_rows)} days through {last_day}, unchanged")


def collect_kalshi():
    try:
        import kalshi
        kalshi.save_price_snapshot()
        kalshi.update_settled()
    except Exception as e:
        print(f"Kalshi failed: {e}")


if __name__ == "__main__":
    collect_dcr()
    collect_booking()
    collect_portwatch()
    collect_kalshi()
