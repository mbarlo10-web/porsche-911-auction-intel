from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

SALES_CSV = BASE_DIR / "data" / "avant_garde_sales_2023_2025.csv"
LISTINGS_CSV = BASE_DIR / "data" / "avant_garde_listings_raw.csv"
OUTPUT_CSV = BASE_DIR / "data" / "avant_garde_sales_with_text.csv"


def _norm_str(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace({"NAN": ""})
    )


def main() -> None:
    if not SALES_CSV.exists():
        raise FileNotFoundError(f"Sales CSV not found at {SALES_CSV}")
    if not LISTINGS_CSV.exists():
        raise FileNotFoundError(f"Listings CSV not found at {LISTINGS_CSV}")

    sales = pd.read_csv(SALES_CSV)
    listings = pd.read_csv(LISTINGS_CSV)

    # Normalize join keys
    for df, vin_col, stock_col in [
        (sales, "vin", "stock_number"),
        (listings, "vin", "stock_number"),
    ]:
        if vin_col in df.columns:
            df["_vin_key"] = _norm_str(df[vin_col])
        else:
            df["_vin_key"] = ""
        if stock_col in df.columns:
            df["_stock_key"] = _norm_str(df[stock_col])
        else:
            df["_stock_key"] = ""

    # Prefer VIN-based join; fall back to stock number where VIN missing/ambiguous.
    sales_vin = sales[sales["_vin_key"] != ""].copy()
    sales_no_vin = sales[sales["_vin_key"] == ""].copy()

    merged_vin = pd.merge(
        sales_vin,
        listings,
        how="left",
        left_on="_vin_key",
        right_on="_vin_key",
        suffixes=("", "_listing"),
    )

    merged_stock = pd.merge(
        sales_no_vin,
        listings,
        how="left",
        left_on="_stock_key",
        right_on="_stock_key",
        suffixes=("", "_listing"),
    )

    merged = pd.concat([merged_vin, merged_stock], ignore_index=True)

    # ------------------------------------------------------------------
    # Fuzzy matching fallback: year + model + nearest mileage
    # ------------------------------------------------------------------
    # Build helper views of listings keyed by normalized model/year.
    listings_copy = listings.copy()
    listings_copy["model_norm"] = (
        listings_copy.get("model_listing", listings_copy.get("model", ""))
        .astype(str)
        .str.upper()
        .str.replace(r"\\s+", " ", regex=True)
        .str.strip()
    )
    listings_copy["year_int"] = pd.to_numeric(
        listings_copy.get("year_listing", listings_copy.get("year", pd.NA)),
        errors="coerce",
    )

    merged["model_norm"] = (
        merged["model"].astype(str).str.upper().str.replace(r"\\s+", " ", regex=True).str.strip()
    )
    merged["year_int"] = pd.to_numeric(merged["year"], errors="coerce")

    # Recompute still-unmatched after adding helper columns so they are present.
    no_desc_mask = merged["description"].isna()
    still_unmatched = merged[no_desc_mask].copy()

    used_listing_idxs = set()

    for idx, row in still_unmatched.iterrows():
        y = row["year_int"]
        m = row["model_norm"]
        odometer = row.get("mileage", None)

        if pd.isna(y) or not m or pd.isna(odometer):
            continue

        # Candidate listings: same year and model string contains core token(s).
        cand = listings_copy[
            (listings_copy["year_int"] == y)
            & (listings_copy["model_norm"].str.contains(m.split()[0], na=False))
        ].copy()

        if cand.empty:
            continue

        # Compute mileage distance
        cand["miles_int"] = pd.to_numeric(cand.get("miles", pd.NA), errors="coerce")
        cand = cand[cand["miles_int"].notna()]
        if cand.empty:
            continue

        cand["mile_diff"] = (cand["miles_int"] - odometer).abs()
        best = cand.sort_values("mile_diff").iloc[0]

        # Require reasonably close odometer (< 10k miles difference).
        if best["mile_diff"] > 10000:
            continue

        if best.name in used_listing_idxs:
            continue
        used_listing_idxs.add(best.name)

        # Attach listing fields to this merged row.
        merged.loc[idx, "listing_url"] = best.get("listing_url")
        merged.loc[idx, "title"] = best.get("title")
        merged.loc[idx, "year_listing"] = best.get("year")
        merged.loc[idx, "make_listing"] = best.get("make")
        merged.loc[idx, "model_listing"] = best.get("model")
        merged.loc[idx, "miles"] = best.get("miles")
        merged.loc[idx, "description"] = best.get("description")

    # Keep a tidy subset plus raw text fields.
    keep_cols = [
        "sale_date",
        "stock_number",
        "vehicle_age_days",
        "year",
        "make",
        "model",
        "mileage",
        "sold_price",
        "body_style",
        "color",
        "company_number",
        "location_label",
        "vin",
        "new_used",
        "listing_url",
        "title",
        "year_listing",
        "make_listing",
        "model_listing",
        "miles",
        "description",
    ]

    # Rename some listing-side cols to avoid confusion.
    if "year_y" in merged.columns:
        merged = merged.rename(columns={"year_y": "year_listing"})
    if "make_y" in merged.columns:
        merged = merged.rename(columns={"make_y": "make_listing"})
    if "model_y" in merged.columns:
        merged = merged.rename(columns={"model_y": "model_listing"})

    if "year_listing" not in merged.columns:
        merged["year_listing"] = pd.NA
    if "make_listing" not in merged.columns:
        merged["make_listing"] = pd.NA
    if "model_listing" not in merged.columns:
        merged["model_listing"] = pd.NA

    # Ensure listing_url, miles, description present
    for col in ["listing_url", "miles", "description"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    out = merged[[c for c in keep_cols if c in merged.columns]].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    matched = out["description"].notna().sum()
    print(f"Total sales rows: {len(out):,}")
    print(f"Rows with matched listing description: {matched:,}")
    print(f"Saved joined dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

