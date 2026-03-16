from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "data" / "avant_garde_sales_with_text.csv"
OUTPUT_CSV = BASE_DIR / "data" / "avant_garde_sales_with_nlp.csv"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Use description as listing_details; some rows will be NaN.
    df["listing_details"] = df["description"]

    text = df["listing_details"].fillna("").astype(str)
    df["listing_details_clean"] = (
        text.str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Simple keyword-based NLP flags (same as app/BaT model).
    t = df["listing_details_clean"]

    df["manual_transmission"] = t.str.contains("manual", na=False).astype(int)
    df["turbo_engine"] = t.str.contains(
        r"turbocharged|twin-turbo|turbo engine",
        regex=True,
        na=False,
    ).astype(int)
    df["limited_slip_diff"] = t.str.contains("limited-slip", na=False).astype(int)
    df["service_records"] = t.str.contains("service records", na=False).astype(int)
    df["clean_carfax"] = t.str.contains("clean carfax", na=False).astype(int)
    df["sport_package"] = t.str.contains("sport chrono", na=False).astype(int)

    df["ceramic_brakes"] = t.str.contains(
        r"ceramic brakes|pccb|carbon-ceramic",
        regex=True,
        na=False,
    ).astype(int)
    df["pdk_transmission"] = t.str.contains("pdk", na=False).astype(int)
    df["bucket_seats"] = t.str.contains(
        r"bucket seats|full bucket seats",
        regex=True,
        na=False,
    ).astype(int)

    df["color_guards_red"] = t.str.contains("guards red", na=False).astype(int)
    df["color_grand_prix_white"] = t.str.contains("grand prix white", na=False).astype(int)
    df["color_speed_yellow"] = t.str.contains("speed yellow", na=False).astype(int)
    df["rare_color"] = (
        df["color_guards_red"]
        | df["color_grand_prix_white"]
        | df["color_speed_yellow"]
    ).astype(int)

    # Low mileage based on odometer in this AG dataset.
    df["low_mileage"] = (df["mileage"] < 30000).astype(int)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Rows in input: {len(df):,}")
    print(f"Saved with NLP features to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

