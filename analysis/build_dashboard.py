"""
build_dashboard.py
Processes SETC Deepavali data and generates a self-contained dashboard.
Output: output/dashboard.html  (open directly in any browser, no server needed)
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "FINAL_deepavali Festival_2021-2025@06062026 1309HRS.xlsx")
OUTPUT_DIR = os.path.join(ROOT, "output")
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

FESTIVAL_DATES = {
    2021: pd.Timestamp("2021-11-04"),
    2022: pd.Timestamp("2022-10-24"),
    2023: pd.Timestamp("2023-11-12"),
    2024: pd.Timestamp("2024-10-31"),
    2025: pd.Timestamp("2025-10-20"),
}
FORECAST_YEAR = 2026
ALL_YEARS = [2021, 2022, 2023, 2024, 2025]
FIT_YEARS  = [2021, 2022, 2023, 2024, 2025]
# Weight by recency: 2021→1, 2022→2, ..., 2025→5
FIT_WEIGHTS = {y: (y - 2020) for y in FIT_YEARS}

PERIOD_ORDER = ["U4", "U3", "U2", "U1", "F", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]
PERIOD_LABELS = {
    "U4": "4 wks before", "U3": "3 wks before", "U2": "2 wks before",
    "U1": "1 wk before",  "F":  "Deepavali Day",
    "Z1": "1 wk after",   "Z2": "2 wks after",  "Z3": "3 wks after",
    "Z4": "4 wks after",  "Z5": "5 wks after",  "Z6": "6 wks after", "Z7": "7 wks after",
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else round(float(obj), 2)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        return super().default(obj)


def lr_fit(pairs, target):
    """Weighted trend forecast: recent years count more (2021=1x … 2025=5x).
    Falls back to last known value when fewer than 2 data points."""
    if len(pairs) == 0:
        return 0
    if len(pairs) == 1:
        return int(pairs[0][1])
    X = np.array([p[0] for p in pairs]).reshape(-1, 1)
    y = np.array([p[1] for p in pairs], dtype=float)
    w = np.array([FIT_WEIGHTS.get(p[0], 1) for p in pairs], dtype=float)
    pred = LinearRegression().fit(X, y, sample_weight=w).predict([[target]])[0]
    return int(max(0, round(pred)))


# ─── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading data …")
    df = pd.read_excel(DATA_FILE, sheet_name="Final", engine="openpyxl")
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df["Corp Code"] = df["Corp Code"].replace({"Tirunelvel": "Tirunelveli", "TIRUNELVEL": "TIRUNELVELI"})
    df["Valid"]    = df["Booked Seats"] <= df["Total Seat"]
    df["Occ"]      = (df["Booked Seats"] / df["Total Seat"] * 100).round(1)
    df["Route"]    = df["New From Place"] + " → " + df["New To Place"]
    df["DOJ"]      = pd.to_datetime(df["DOJ"])
    df["Day_Offset"] = df.apply(
        lambda r: int((r["DOJ"] - FESTIVAL_DATES[r["Year"]]).days)
        if r["Year"] in FESTIVAL_DATES else None, axis=1
    )
    df["Dept_Hour"] = df["Dept Time"].apply(
        lambda t: int(str(t).split(":")[0]) if pd.notna(t) and ":" in str(t) else None
    )
    print(f"  {len(df):,} records loaded")
    return df


# ─── Yearly trend ─────────────────────────────────────────────────────────────
def build_yearly(df):
    v = df[df["Valid"]]
    rows = []
    for y in ALL_YEARS:
        g = v[v["Year"] == y]
        rows.append({
            "year": int(y),
            "services":   int(len(g)),
            "passengers": int(g["Booked Seats"].sum()),
            "occupancy":  round(float(g["Occ"].mean()), 1),
            "is_forecast": False,
        })
    fit_s = [(r["year"], r["services"])   for r in rows if r["year"] in FIT_YEARS]
    fit_p = [(r["year"], r["passengers"]) for r in rows if r["year"] in FIT_YEARS]
    rows.append({
        "year": FORECAST_YEAR,
        "services":   lr_fit(fit_s, FORECAST_YEAR),
        "passengers": lr_fit(fit_p, FORECAST_YEAR),
        "occupancy":  None,
        "is_forecast": True,
    })
    return rows


# ─── Day-level (D-14 … D+14) ──────────────────────────────────────────────────
def build_day_level(df):
    v = df[df["Valid"] & df["Day_Offset"].notna()]
    v = v[v["Day_Offset"].between(-14, 14)]
    grp = v.groupby(["Year", "Day_Offset"]).agg(
        services=("Trip Code", "count"),
        passengers=("Booked Seats", "sum"),
    ).reset_index()

    result = []
    for offset in range(-14, 15):
        label = "D-Day" if offset == 0 else f"D{offset:+d}"
        entry = {"offset": int(offset), "label": label}
        fit_s, fit_p = [], []
        for y in ALL_YEARS:
            row = grp[(grp["Year"] == y) & (grp["Day_Offset"] == offset)]
            s = int(row["services"].values[0])   if len(row) else None
            p = int(row["passengers"].values[0]) if len(row) else None
            entry[f"s{y}"] = s
            entry[f"p{y}"] = p
            if s is not None and y in FIT_YEARS: fit_s.append((y, s))
            if p is not None and y in FIT_YEARS: fit_p.append((y, p))
        entry[f"s{FORECAST_YEAR}"] = lr_fit(fit_s, FORECAST_YEAR) if len(fit_s) >= 2 else None
        entry[f"p{FORECAST_YEAR}"] = lr_fit(fit_p, FORECAST_YEAR) if len(fit_p) >= 2 else None
        result.append(entry)
    return result


# ─── Period analysis ──────────────────────────────────────────────────────────
def build_periods(df):
    v = df[df["Valid"]]
    grp = v.groupby(["Period", "Year"]).agg(
        services=("Trip Code", "count"),
        passengers=("Booked Seats", "sum"),
    ).reset_index()

    result = []
    for period in PERIOD_ORDER:
        entry = {"period": period, "label": PERIOD_LABELS.get(period, period)}
        fit_s, fit_p = [], []
        for y in ALL_YEARS:
            row = grp[(grp["Period"] == period) & (grp["Year"] == y)]
            sv = int(row["services"].values[0])   if len(row) else None
            pv = int(row["passengers"].values[0]) if len(row) else None
            entry[f"s{y}"] = sv
            entry[f"p{y}"] = pv
            if sv is not None and y in FIT_YEARS: fit_s.append((y, sv))
            if pv is not None and y in FIT_YEARS: fit_p.append((y, pv))
        entry[f"s{FORECAST_YEAR}"] = lr_fit(fit_s, FORECAST_YEAR)
        entry[f"p{FORECAST_YEAR}"] = lr_fit(fit_p, FORECAST_YEAR)
        result.append(entry)
    return result


# ─── Top routes ───────────────────────────────────────────────────────────────
def build_routes(df, n=20):
    v = df[df["Valid"]]
    totals = v.groupby("Route")["Booked Seats"].sum().nlargest(n)
    sub  = v[v["Route"].isin(totals.index)]

    # Dominant corp per route (most services)
    corp_counts = sub.groupby(["Route", "Corp Code"])["Trip Code"].count().reset_index()
    corp_counts = corp_counts.sort_values("Trip Code", ascending=False)
    dominant_corp = corp_counts.groupby("Route")["Corp Code"].first().to_dict()
    all_route_corps = (
        sub.groupby("Route")["Corp Code"].unique()
        .apply(lambda x: sorted(x.tolist())).to_dict()
    )

    grp  = sub.groupby(["Route", "Year"]).agg(
        services=("Trip Code", "count"),
        passengers=("Booked Seats", "sum"),
    ).reset_index()

    result = []
    for route in totals.index:
        parts = route.split(" → ")
        entry = {
            "route":         route,
            "from_place":    parts[0] if len(parts) > 0 else "",
            "to_place":      parts[1] if len(parts) > 1 else "",
            "total_5yr":     int(totals[route]),
            "dominant_corp": dominant_corp.get(route, ""),
            "all_corps":     all_route_corps.get(route, []),
        }
        fit_s, fit_p = [], []
        for y in ALL_YEARS:
            row = grp[(grp["Route"] == route) & (grp["Year"] == y)]
            sv = int(row["services"].values[0])   if len(row) else None
            pv = int(row["passengers"].values[0]) if len(row) else None
            entry[f"s{y}"] = sv
            entry[f"p{y}"] = pv
            if sv is not None and y in FIT_YEARS: fit_s.append((y, sv))
            if pv is not None and y in FIT_YEARS: fit_p.append((y, pv))
        entry["s2026"] = lr_fit(fit_s, FORECAST_YEAR)
        entry["p2026"] = lr_fit(fit_p, FORECAST_YEAR)
        n_yrs = len(fit_s)
        if n_yrs >= 4:   entry["forecast_basis"] = f"{n_yrs}-yr trend"
        elif n_yrs >= 2: entry["forecast_basis"] = f"{n_yrs}-yr trend ⚠"
        else:            entry["forecast_basis"] = "2024 data only ⚠"
        result.append(entry)
    return sorted(result, key=lambda x: x["p2026"], reverse=True)


# ─── Occupancy by route ───────────────────────────────────────────────────────
def build_occupancy(df):
    v = df[df["Valid"]]
    grp = v.groupby("Route").agg(
        occ=("Occ", "mean"),
        services=("Trip Code", "count"),
        passengers=("Booked Seats", "sum"),
    ).reset_index()
    grp = grp[grp["services"] >= 50].nlargest(20, "passengers")
    grp["occ"]    = grp["occ"].round(1)
    grp["status"] = grp["occ"].apply(
        lambda x: "high" if x >= 70 else ("moderate" if x >= 50 else "low")
    )
    return grp[["Route", "occ", "services", "passengers", "status"]].rename(
        columns={"Route": "route"}
    ).to_dict("records")


# ─── Departure hours ──────────────────────────────────────────────────────────
def build_departure_hours(df):
    v = df[df["Valid"] & df["Dept_Hour"].notna()]
    v = v[v["Dept_Hour"].between(4, 23)]
    grp = v.groupby("Dept_Hour")["Trip Code"].count().reset_index()
    grp.columns = ["hour", "services"]
    grp = grp.sort_values("hour")
    grp["label"] = grp["hour"].apply(lambda h: f"{int(h):02d}:00")
    return grp.to_dict("records")


# ─── Bus class ────────────────────────────────────────────────────────────────
def build_bus_class(df):
    v = df[df["Valid"]]
    totals = v.groupby("New Class Name")["Trip Code"].count().nlargest(8)
    return [{"name": str(k), "value": int(val)} for k, val in totals.items()]


# ─── Corporations ─────────────────────────────────────────────────────────────
def build_corps(df):
    v = df[df["Valid"]]
    grp = v.groupby(["Corp Code", "Year"])["Trip Code"].count().reset_index()
    grp.columns = ["corp", "year", "services"]
    corps = v.groupby("Corp Code")["Trip Code"].count().nlargest(8).index.tolist()
    result = []
    for c in corps:
        entry = {"corp": c}
        fit_s = []
        for y in ALL_YEARS:
            row = grp[(grp["corp"] == c) & (grp["year"] == y)]
            val = int(row["services"].values[0]) if len(row) else 0
            entry[f"s{y}"] = val
            if y in FIT_YEARS: fit_s.append((y, val))
        entry["s2026"] = lr_fit(fit_s, FORECAST_YEAR) if len(fit_s) >= 2 else 0
        result.append(entry)
    return result


# ─── Sectors ─────────────────────────────────────────────────────────────────
def build_sectors(df):
    v = df[df["Valid"]]
    grp = v.groupby(["Sector", "Year"])["Trip Code"].count().reset_index()
    grp.columns = ["sector", "year", "services"]
    sectors = sorted(v["Sector"].unique())
    result = []
    for s in sectors:
        entry = {"sector": s}
        fit_s = []
        for y in ALL_YEARS:
            row = grp[(grp["sector"] == s) & (grp["year"] == y)]
            val = int(row["services"].values[0]) if len(row) else 0
            entry[f"s{y}"] = val
            if y in FIT_YEARS: fit_s.append((y, val))
        entry["s2026"] = lr_fit(fit_s, FORECAST_YEAR) if len(fit_s) >= 2 else 0
        result.append(entry)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
# ─── Per-corp breakdowns (for corp filter across all tabs) ────────────────────
def build_corp_breakdowns(df, corps_list):
    by_corp = {}
    for corp in corps_list:
        cdf = df[df["Corp Code"] == corp].copy()
        by_corp[corp] = {
            "yearly":    build_yearly(cdf),
            "day_level": build_day_level(cdf),
            "periods":   build_periods(cdf),
        }
    return by_corp


if __name__ == "__main__":
    df = load_data()

    print("Yearly trend …");          yearly    = build_yearly(df)
    print("Day-level D-14…D+14 …");   day_level = build_day_level(df)
    print("Period analysis …");       periods   = build_periods(df)
    print("Top routes …");            routes    = build_routes(df)
    print("Occupancy …");             occupancy = build_occupancy(df)
    print("Departure hours …");       dept_hrs  = build_departure_hours(df)
    print("Bus class …");             bus_class = build_bus_class(df)
    print("Corporations …");          corps     = build_corps(df)
    print("Sectors …");               sectors   = build_sectors(df)
    print("Per-corp breakdowns …")
    corps_list = [c["corp"] for c in corps]
    by_corp    = build_corp_breakdowns(df, corps_list)

    forecast_row = next(r for r in yearly if r["year"] == FORECAST_YEAR)
    top_route    = routes[0] if routes else {}
    kpis = {
        "services_2026":        forecast_row["services"],
        "passengers_2026":      forecast_row["passengers"],
        "festival_date":        "October 20, 2026",
        "top_route":            top_route.get("route", ""),
        "top_route_passengers": top_route.get("p2026", 0),
    }

    data = {
        "kpis":       kpis,
        "yearly":     yearly,
        "day_level":  day_level,
        "periods":    periods,
        "routes":     routes,
        "occupancy":  occupancy,
        "dept_hours": dept_hrs,
        "bus_class":  bus_class,
        "corps":      corps,
        "sectors":    sectors,
        "by_corp":    by_corp,
        "filter_options": {
            "corps":       sorted(list(set(r["dominant_corp"] for r in routes if r["dominant_corp"]))),
            "from_places": sorted(list(set(r["from_place"] for r in routes if r["from_place"]))),
            "to_places":   sorted(list(set(r["to_place"]   for r in routes if r["to_place"]))),
        },
        "meta": {
            "generated_on":    "2026-06-07",
            "festival_date":   "2026-10-20",
            "forecast_method": "OLS Linear Regression, fitted on 2021–2024 data",
            "source":          "SETC Deepavali Festival Bookings 2021–2025",
            "note":            "2025 excluded from model fitting — low occupancy (14%) indicates unconfirmed advance bookings",
        },
    }

    # Save companion JSON
    json_path = os.path.join(OUTPUT_DIR, "dashboard_data.json")
    with open(json_path, "w") as f:
        json.dump(data, f, cls=NpEncoder, indent=2)
    print(f"\nSaved: {json_path}")

    # Inject data into HTML template
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = f.read()

    json_str = json.dumps(data, cls=NpEncoder)
    json_str  = json_str.replace("</script>", r"<\/script>")
    html = template.replace("__DASHBOARD_DATA__", json_str)

    html_path = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path}")
    print("\nDone!  Open  output/dashboard.html  in your browser.")
