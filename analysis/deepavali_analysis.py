"""
Deepavali 2026 Demand Forecast
Analyzes 2021-2025 SETC festival booking data and forecasts 2026 demand.
Outputs: multi-sheet Excel + companion CSVs for NotebookLM.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data",
                         "FINAL_deepavali Festival_2021-2025@06062026 1309HRS.xlsx")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEEPAVALI_2026_DATE = "2026-10-20"  # Approximate; adjust if official date differs
FORECAST_YEAR = 2026

PERIOD_ORDER = ["U4", "U3", "U2", "U1", "F", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]
PERIOD_LABELS = {
    "U4": "Before (-4 wk)", "U3": "Before (-3 wk)", "U2": "Before (-2 wk)",
    "U1": "Before (-1 wk)", "F": "Festival Day",
    "Z1": "After (+1 wk)", "Z2": "After (+2 wk)", "Z3": "After (+3 wk)",
    "Z4": "After (+4 wk)", "Z5": "After (+5 wk)", "Z6": "After (+6 wk)", "Z7": "After (+7 wk)",
}

FESTIVAL_DATES = {
    2021: "2021-11-04", 2022: "2022-10-24", 2023: "2023-11-12",
    2024: "2024-10-31", 2025: "2025-10-20",
}


# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
def load_data(path):
    print("Loading data...")
    df = pd.read_excel(path, sheet_name="Final", engine="openpyxl")
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Strip string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Normalise duplicate corp name
    df["Corp Code"] = df["Corp Code"].replace("Tirunelvel", "Tirunelveli")

    # Occupancy
    df["Occupancy_Pct"] = (df["Booked Seats"] / df["Total Seat"] * 100).round(2)

    # Clean df for passenger calculations (exclude >100% occupancy anomalies)
    df["Valid_Booking"] = df["Booked Seats"] <= df["Total Seat"]

    # Route label
    df["Route"] = df["New From Place"] + " → " + df["New To Place"]

    # Period as ordered category
    df["Period"] = pd.Categorical(df["Period"], categories=PERIOD_ORDER, ordered=True)
    df["Period_Label"] = df["Period"].map(PERIOD_LABELS)

    print(f"  Years present: {sorted(df['Year'].unique())}")
    print(f"  Periods present: {sorted(df['Period'].dropna().unique())}")
    return df


# ---------------------------------------------------------------------------
# 2. Linear regression forecast helper
# ---------------------------------------------------------------------------
def forecast_series(years: list, values: list, target_year: int,
                    fit_years=None) -> float:
    """Fit LR on fit_years (default all) and predict target_year."""
    pairs = [(y, v) for y, v in zip(years, values) if not np.isnan(v)]
    if fit_years:
        pairs = [(y, v) for y, v in pairs if y in fit_years]
    if len(pairs) < 2:
        return np.nan
    X = np.array([p[0] for p in pairs]).reshape(-1, 1)
    y = np.array([p[1] for p in pairs])
    model = LinearRegression().fit(X, y)
    pred = model.predict([[target_year]])[0]
    return max(0.0, round(pred, 1))


# ---------------------------------------------------------------------------
# 3. Historical aggregation
# ---------------------------------------------------------------------------
def yearly_summary(df):
    valid = df[df["Valid_Booking"]]
    grp = df.groupby("Year")
    grp_v = valid.groupby("Year")
    result = pd.DataFrame({
        "Year": sorted(df["Year"].unique()),
    })
    result["Total_Services"] = result["Year"].map(grp["Trip Code"].count())
    result["Total_Passengers"] = result["Year"].map(grp_v["Booked Seats"].sum())
    result["Total_Capacity"] = result["Year"].map(grp_v["Total Seat"].sum())
    result["Avg_Occupancy_Pct"] = (result["Total_Passengers"] / result["Total_Capacity"] * 100).round(2)
    result["Festival_Date"] = result["Year"].map(FESTIVAL_DATES)
    return result


def period_trend(df):
    valid = df[df["Valid_Booking"]]
    grp = df.groupby(["Year", "Period"], observed=True)
    grp_v = valid.groupby(["Year", "Period"], observed=True)

    svc = grp["Trip Code"].count().rename("Services").reset_index()
    pax = grp_v["Booked Seats"].sum().rename("Passengers").reset_index()
    cap = grp_v["Total Seat"].sum().rename("Capacity").reset_index()

    merged = svc.merge(pax, on=["Year", "Period"]).merge(cap, on=["Year", "Period"])
    merged["Occupancy_Pct"] = (merged["Passengers"] / merged["Capacity"] * 100).round(2)
    merged["Period_Label"] = merged["Period"].map(PERIOD_LABELS)

    # Pivot for readability
    svc_pivot = merged.pivot(index="Period", columns="Year", values="Services").reindex(PERIOD_ORDER)
    pax_pivot = merged.pivot(index="Period", columns="Year", values="Passengers").reindex(PERIOD_ORDER)

    years = sorted(df["Year"].unique())
    fit_years = [y for y in years if y <= 2024]

    # Forecast 2026 per period
    svc_2026, pax_2026 = [], []
    for period in PERIOD_ORDER:
        row = merged[merged["Period"] == period]
        s_vals = [row[row["Year"] == y]["Services"].values[0]
                  if len(row[row["Year"] == y]) else np.nan for y in years]
        p_vals = [row[row["Year"] == y]["Passengers"].values[0]
                  if len(row[row["Year"] == y]) else np.nan for y in years]
        sv = forecast_series(years, s_vals, FORECAST_YEAR, fit_years)
        pv = forecast_series(years, p_vals, FORECAST_YEAR, fit_years)
        svc_2026.append(0 if np.isnan(sv) else int(round(sv)))
        pax_2026.append(0 if np.isnan(pv) else int(round(pv)))

    svc_pivot[FORECAST_YEAR] = svc_2026
    pax_pivot[FORECAST_YEAR] = pax_2026

    svc_pivot.index = [PERIOD_LABELS.get(p, p) for p in svc_pivot.index]
    pax_pivot.index = [PERIOD_LABELS.get(p, p) for p in pax_pivot.index]

    svc_pivot.columns = [str(c) for c in svc_pivot.columns]
    pax_pivot.columns = [str(c) for c in pax_pivot.columns]

    # Combined flat table for CSV
    flat = merged.copy()
    flat["Period"] = flat["Period"].astype(str)
    forecast_rows = pd.DataFrame({
        "Year": FORECAST_YEAR,
        "Period": PERIOD_ORDER,
        "Services": svc_2026,
        "Passengers": pax_2026,
        "Capacity": [np.nan] * len(PERIOD_ORDER),
        "Occupancy_Pct": [np.nan] * len(PERIOD_ORDER),
        "Period_Label": [PERIOD_LABELS[p] for p in PERIOD_ORDER],
    })
    flat = pd.concat([flat, forecast_rows], ignore_index=True)

    return svc_pivot, pax_pivot, flat


# ---------------------------------------------------------------------------
# 4. Route analysis
# ---------------------------------------------------------------------------
def route_analysis(df, top_n=40):
    valid = df[df["Valid_Booking"]]
    years = sorted(df["Year"].unique())
    fit_years = [y for y in years if y <= 2024]

    # Top 40 routes by historical total passengers
    route_totals = valid.groupby("Route")["Booked Seats"].sum().nlargest(top_n)
    top_routes = route_totals.index.tolist()

    sub = valid[valid["Route"].isin(top_routes)]

    # Year × Route
    yr_route = sub.groupby(["Route", "Year"]).agg(
        Services=("Trip Code", "count"),
        Passengers=("Booked Seats", "sum"),
    ).reset_index()

    # Historical wide table
    hist = yr_route.pivot_table(index="Route", columns="Year",
                                values=["Services", "Passengers"]).fillna(0).astype(int)
    hist.columns = [f"{m}_{y}" for m, y in hist.columns]
    hist = hist.reindex(top_routes)
    hist["Total_Passengers_5yr"] = route_totals.reindex(hist.index).values

    # Forecast 2026
    svc_f, pax_f = [], []
    for route in top_routes:
        row = yr_route[yr_route["Route"] == route]
        s_vals = [row[row["Year"] == y]["Services"].values[0]
                  if len(row[row["Year"] == y]) else 0 for y in years]
        p_vals = [row[row["Year"] == y]["Passengers"].values[0]
                  if len(row[row["Year"] == y]) else 0 for y in years]
        sv = forecast_series(years, s_vals, FORECAST_YEAR, fit_years)
        pv = forecast_series(years, p_vals, FORECAST_YEAR, fit_years)
        svc_f.append(0 if np.isnan(sv) else int(round(sv)))
        pax_f.append(0 if np.isnan(pv) else int(round(pv)))

    hist[f"Services_{FORECAST_YEAR}_Forecast"] = svc_f
    hist[f"Passengers_{FORECAST_YEAR}_Forecast"] = pax_f

    # Avg annual growth rate (CAGR 2021→2024)
    def safe_cagr(start, end, n=3):
        if start <= 0 or end <= 0:
            return np.nan
        return round(((end / start) ** (1 / n) - 1) * 100, 1)

    hist["Passengers_CAGR_2021_24_Pct"] = [
        safe_cagr(
            yr_route[(yr_route["Route"] == r) & (yr_route["Year"] == 2021)]["Passengers"].values[0]
            if len(yr_route[(yr_route["Route"] == r) & (yr_route["Year"] == 2021)]) else 0,
            yr_route[(yr_route["Route"] == r) & (yr_route["Year"] == 2024)]["Passengers"].values[0]
            if len(yr_route[(yr_route["Route"] == r) & (yr_route["Year"] == 2024)]) else 0,
        ) for r in top_routes
    ]
    hist = hist.reset_index()

    # Route × Period 2026 forecast
    sub_period = sub.groupby(["Route", "Period", "Year"], observed=True).agg(
        Services=("Trip Code", "count"),
        Passengers=("Booked Seats", "sum"),
    ).reset_index()

    route_period_rows = []
    for route in top_routes:
        for period in PERIOD_ORDER:
            rp = sub_period[(sub_period["Route"] == route) & (sub_period["Period"] == period)]
            s_vals = [rp[rp["Year"] == y]["Services"].values[0]
                      if len(rp[rp["Year"] == y]) else 0 for y in years]
            p_vals = [rp[rp["Year"] == y]["Passengers"].values[0]
                      if len(rp[rp["Year"] == y]) else 0 for y in years]
            sv = forecast_series(years, s_vals, FORECAST_YEAR, fit_years)
            pv = forecast_series(years, p_vals, FORECAST_YEAR, fit_years)
            route_period_rows.append({
                "Route": route,
                "Period": period,
                "Period_Label": PERIOD_LABELS[period],
                "Services_2026_Forecast": 0 if np.isnan(sv) else int(round(sv)),
                "Passengers_2026_Forecast": 0 if np.isnan(pv) else int(round(pv)),
            })
    route_period_df = pd.DataFrame(route_period_rows)

    return hist, route_period_df


# ---------------------------------------------------------------------------
# 5. Sector & corp analysis
# ---------------------------------------------------------------------------
def sector_analysis(df):
    valid = df[df["Valid_Booking"]]
    years = sorted(df["Year"].unique())
    fit_years = [y for y in years if y <= 2024]

    grp = valid.groupby(["Sector", "Year", "Period"], observed=True).agg(
        Services=("Trip Code", "count"),
        Passengers=("Booked Seats", "sum"),
    ).reset_index()
    grp["Period"] = grp["Period"].astype(str)

    # 2026 forecast per sector × period
    forecast_rows = []
    for sector in grp["Sector"].unique():
        for period in PERIOD_ORDER:
            sub = grp[(grp["Sector"] == sector) & (grp["Period"] == period)]
            s_vals = [sub[sub["Year"] == y]["Services"].values[0]
                      if len(sub[sub["Year"] == y]) else 0 for y in years]
            p_vals = [sub[sub["Year"] == y]["Passengers"].values[0]
                      if len(sub[sub["Year"] == y]) else 0 for y in years]
            sv = forecast_series(years, s_vals, FORECAST_YEAR, fit_years)
            pv = forecast_series(years, p_vals, FORECAST_YEAR, fit_years)
            forecast_rows.append({
                "Sector": sector,
                "Year": FORECAST_YEAR,
                "Period": period,
                "Period_Label": PERIOD_LABELS[period],
                "Services": 0 if np.isnan(sv) else int(round(sv)),
                "Passengers": 0 if np.isnan(pv) else int(round(pv)),
            })
    return pd.concat([grp, pd.DataFrame(forecast_rows)], ignore_index=True)


def corp_analysis(df):
    valid = df[df["Valid_Booking"]]
    years = sorted(df["Year"].unique())
    fit_years = [y for y in years if y <= 2024]

    grp = valid.groupby(["Corp Code", "Year"]).agg(
        Services=("Trip Code", "count"),
        Passengers=("Booked Seats", "sum"),
    ).reset_index()

    pivot_svc = grp.pivot(index="Corp Code", columns="Year", values="Services").fillna(0).astype(int)
    pivot_pax = grp.pivot(index="Corp Code", columns="Year", values="Passengers").fillna(0).astype(int)

    corps = grp["Corp Code"].unique()
    svc_f, pax_f = [], []
    for corp in pivot_svc.index:
        s_vals = [int(pivot_svc.loc[corp, y]) if y in pivot_svc.columns else 0 for y in years]
        p_vals = [int(pivot_pax.loc[corp, y]) if y in pivot_pax.columns else 0 for y in years]
        sv = forecast_series(years, s_vals, FORECAST_YEAR, fit_years)
        pv = forecast_series(years, p_vals, FORECAST_YEAR, fit_years)
        svc_f.append(0 if np.isnan(sv) else int(round(sv)))
        pax_f.append(0 if np.isnan(pv) else int(round(pv)))

    pivot_svc[f"{FORECAST_YEAR}_Forecast"] = svc_f
    pivot_pax[f"{FORECAST_YEAR}_Forecast"] = pax_f
    pivot_svc.columns = [str(c) for c in pivot_svc.columns]
    pivot_pax.columns = [str(c) for c in pivot_pax.columns]

    result = pivot_svc.add_prefix("Services_").join(pivot_pax.add_prefix("Passengers_")).reset_index()
    return result


def busclass_analysis(df):
    valid = df[df["Valid_Booking"]]
    grp = valid.groupby(["New Class Name", "Year"]).agg(
        Services=("Trip Code", "count"),
        Passengers=("Booked Seats", "sum"),
    ).reset_index()

    svc_pivot = grp.pivot_table(index="New Class Name", columns="Year",
                                values="Services", fill_value=0).astype(int)
    svc_pivot.columns = [str(c) for c in svc_pivot.columns]

    # Share % per year
    share = svc_pivot.div(svc_pivot.sum(axis=0), axis=1).mul(100).round(1)
    share.columns = [f"Share_{c}_Pct" for c in share.columns]

    result = svc_pivot.join(share).reset_index()
    return result


# ---------------------------------------------------------------------------
# 6. Summary sheet
# ---------------------------------------------------------------------------
def build_summary(yearly_df, period_flat_df):
    years = [y for y in yearly_df["Year"] if y <= 2024]
    s_vals = yearly_df[yearly_df["Year"].isin(years)]["Total_Services"].tolist()
    p_vals = yearly_df[yearly_df["Year"].isin(years)]["Total_Passengers"].tolist()

    svc_2026 = int(round(forecast_series(years, s_vals, FORECAST_YEAR)))
    pax_2026 = int(round(forecast_series(years, p_vals, FORECAST_YEAR)))

    pf = period_flat_df[period_flat_df["Year"] == FORECAST_YEAR]
    peak_period = pf.loc[pf["Passengers"].idxmax(), "Period_Label"] if len(pf) else "N/A"

    rows = [
        ("Deepavali 2026 Expected Date", DEEPAVALI_2026_DATE),
        ("", ""),
        ("--- 2026 FORECAST (based on 2021-2024 trend) ---", ""),
        ("Total Services Expected (Full Window)", f"{svc_2026:,}"),
        ("Total Passengers Expected (Full Window)", f"{pax_2026:,}"),
        ("Peak Travel Period", peak_period),
        ("", ""),
        ("--- HISTORICAL REFERENCE ---", ""),
    ]
    for _, row in yearly_df.iterrows():
        rows.append((f"Year {int(row['Year'])} - Total Services", f"{int(row['Total_Services']):,}"))
        rows.append((f"Year {int(row['Year'])} - Total Passengers", f"{int(row['Total_Passengers']):,}"))
        rows.append((f"Year {int(row['Year'])} - Avg Occupancy", f"{row['Avg_Occupancy_Pct']:.1f}%"))
        rows.append((f"Year {int(row['Year'])} - Festival Date", str(row['Festival_Date'])))
        rows.append(("", ""))
    rows += [
        ("--- DATA NOTES ---", ""),
        ("Dataset", "SETC Deepavali Festival bookings 2021-2025"),
        ("Records", "109,143 total trip records"),
        ("2025 Data Caveat", "2025 has higher trip count but very low occupancy (14.3%) — likely partially confirmed bookings; forecast uses 2021-2024 as primary fit years"),
        ("Anomalous Records", "198 records with Booked Seats > Total Seat excluded from passenger/occupancy calculations"),
        ("Forecast Method", "OLS Linear Regression (scikit-learn) per metric, fit on 2021-2024, predicting 2026"),
        ("Period Codes", "U4-U1: 4-1 weeks before Deepavali | F: Festival Day | Z1-Z7: 1-7 weeks after Deepavali"),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# 7. Write outputs
# ---------------------------------------------------------------------------
def write_outputs(summary_df, yearly_df, svc_pivot, pax_pivot, period_flat_df,
                  route_hist_df, route_period_df, sector_df, corp_df, busclass_df):
    xlsx_path = os.path.join(OUTPUT_DIR, "deepavali_2026_forecast.xlsx")
    print(f"\nWriting {xlsx_path} ...")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Summary
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Historical yearly
        yearly_df.to_excel(writer, sheet_name="Historical_Yearly", index=False)

        # Period trend — services pivot
        svc_pivot.to_excel(writer, sheet_name="Period_Services_Trend")

        # Period trend — passengers pivot
        pax_pivot.to_excel(writer, sheet_name="Period_Passengers_Trend")

        # Route historical + 2026 forecast
        route_hist_df.to_excel(writer, sheet_name="Route_Forecast_2026", index=False)

        # Route × Period 2026
        route_period_df.to_excel(writer, sheet_name="Route_Period_2026", index=False)

        # Sector analysis
        sector_df.to_excel(writer, sheet_name="Sector_Analysis", index=False)

        # Corp analysis
        corp_df.to_excel(writer, sheet_name="Corp_Analysis", index=False)

        # Bus class
        busclass_df.to_excel(writer, sheet_name="BusClass_Mix", index=False)

        # Data notes (embedded in summary, also separate)
        notes = pd.DataFrame({
            "Note": [
                "2025 data caveat: inflated trip counts (39,064) with very low occupancy (14.3%) — likely partially confirmed bookings. Primary forecast uses 2021-2024.",
                "198 records with Booked Seats > Total Seat excluded from passenger/occupancy calculations.",
                "Forecast method: OLS Linear Regression per metric (scikit-learn), fit on 2021-2024, predicting 2026.",
                "Period U1 = 1 week before Deepavali (highest pre-festival demand).",
                "Period Z1 = 1 week after Deepavali (highest post-festival demand).",
                f"Deepavali 2026 expected date: {DEEPAVALI_2026_DATE}.",
                "Corp Code 'Tirunelvel' normalized to 'Tirunelveli' in analysis.",
            ]
        })
        notes.to_excel(writer, sheet_name="Data_Notes", index=False)

    # Companion CSVs
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_2026.csv"), index=False)
    period_flat_df[period_flat_df["Year"] == FORECAST_YEAR][
        ["Period", "Period_Label", "Services", "Passengers"]
    ].to_csv(os.path.join(OUTPUT_DIR, "period_forecast_2026.csv"), index=False)
    route_hist_df[[
        "Route",
        "Total_Passengers_5yr",
        f"Services_{FORECAST_YEAR}_Forecast",
        f"Passengers_{FORECAST_YEAR}_Forecast",
        "Passengers_CAGR_2021_24_Pct",
    ]].to_csv(os.path.join(OUTPUT_DIR, "route_forecast_2026.csv"), index=False)

    print("Done. Output files:")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(path) // 1024
        print(f"  {f}  ({size_kb} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(DATA_FILE)

    print("\nBuilding historical yearly summary...")
    yearly_df = yearly_summary(df)
    print(yearly_df[["Year", "Total_Services", "Total_Passengers", "Avg_Occupancy_Pct"]].to_string(index=False))

    print("\nBuilding period trend...")
    svc_pivot, pax_pivot, period_flat_df = period_trend(df)

    print("\nBuilding route analysis (top 40 routes)...")
    route_hist_df, route_period_df = route_analysis(df, top_n=40)
    print(f"  Route table: {route_hist_df.shape[0]} routes × {route_hist_df.shape[1]} columns")

    print("\nBuilding sector analysis...")
    sector_df = sector_analysis(df)

    print("\nBuilding corporation analysis...")
    corp_df = corp_analysis(df)

    print("\nBuilding bus class analysis...")
    busclass_df = busclass_analysis(df)

    print("\nBuilding summary sheet...")
    summary_df = build_summary(yearly_df, period_flat_df)

    write_outputs(
        summary_df, yearly_df, svc_pivot, pax_pivot, period_flat_df,
        route_hist_df, route_period_df, sector_df, corp_df, busclass_df,
    )
