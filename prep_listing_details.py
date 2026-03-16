import os

import pandas as pd


BASE_DIR = os.path.dirname(__file__)

# Source BaT / Kaggle-aligned CSV already used in the existing pipeline.
INPUT_CSV = os.path.join(BASE_DIR, "final_predictions_clean_with_corrected_mileage.csv")

# New output path so we do not overwrite the original training CSV.
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "porsche_processed_with_details.csv")


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if "listing_details" not in df.columns:
        raise KeyError("Expected column 'listing_details' not found in input CSV.")

    # Explicitly preserve the original listing_details column
    # and create a cleaned text version for future NLP-style features.
    listing_details = df["listing_details"].fillna("")

    df["listing_details"] = listing_details
    df["listing_details_clean"] = (
        listing_details.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()

