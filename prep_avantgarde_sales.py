import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

# Source Excel exported from Numbers (lives outside the app folder).
EXCEL_PATH = Path(
    "/Users/markbarlow/Desktop/ASU Grad School/Fall 2025/Avant Garde/"
    "AGC_Test/AGC_Porsche_Sales_2023-2025/AGC_Porsche_Sales_2023-2025.xlsx"
)

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_CSV = OUTPUT_DIR / "avant_garde_sales_2023_2025.csv"


def load_raw_sales() -> pd.DataFrame:
    """
    Load the Avant-Garde sales report from the Numbers-exported Excel file.

    The real data is in the sheet named 'Sheet 1 - porsche_sales_report_' with
    a one-row header offset.
    """
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found at {EXCEL_PATH}")

    xls = pd.ExcelFile(EXCEL_PATH)
    target_sheet = "Sheet 1 - porsche_sales_report_"
    if target_sheet not in xls.sheet_names:
        raise ValueError(f"Expected sheet '{target_sheet}' not found. Sheets: {xls.sheet_names}")

    # Header row is the second row (index 1).
    df = pd.read_excel(EXCEL_PATH, sheet_name=target_sheet, header=1)
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Porsche sales and normalize column names to match the modeling pipeline.
    """
    # Drop completely empty rows if any.
    df = df.dropna(how="all")

    # Filter to Porsche; we'll later filter to 911 vs other models as needed.
    if "Make" in df.columns:
        df = df[df["Make"].astype(str).str.upper().str.contains("PORSCHE")]

    rename_map = {
        "Sale Date": "sale_date",
        "Stock Number": "stock_number",
        "Vehicle Age": "vehicle_age_days",
        "Year": "year",
        "Make": "make",
        "Model": "model",
        "Odometer": "mileage",
        "Retail Price": "sold_price",
        "Body Style": "body_style",
        "Color": "color",
        "Company Number": "company_number",
        "Sale Type": "sale_type",
        "VIN": "vin",
        "New/Used": "new_used",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Basic numeric coercions
    for col in ["year", "mileage", "sold_price", "vehicle_age_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize company number to a simple location flag.
    if "company_number" in df.columns:
        df["company_number"] = df["company_number"].astype(str).str.strip()
        df["location_label"] = df["company_number"].map(
            {
                "AG1": "Portland",
                "AG2": "Scottsdale",
            }
        )

    return df


def main() -> None:
    df_raw = load_raw_sales()
    print(f"Raw rows (all makes): {len(df_raw):,}")

    df_clean = clean_sales(df_raw)
    print(f"Filtered Porsche rows: {len(df_clean):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned sales CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

