import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

DATA_PATH = "final_predictions_clean_with_corrected_mileage.csv"
TARGET = "sold_price"

CAT_FEATURES = ["submodel"]

NUM_FEATURES = [
    "year",
    "mileage",
    "owners",
    "accidents",
    "latitude",
    "longitude",
    "views",
    "watchers",
    "comments",
    "auction_month",
    "auction_dow",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

DOW_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def coerce_numeric_like(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        s = s.replace({"None": None, "nan": None, "NaN": None, "": None})
        extracted = s.str.extract(r"(-?\d+\.?\d*)", expand=False)
        df[c] = pd.to_numeric(extracted, errors="coerce")
    return df


def coerce_month(df: pd.DataFrame, col: str = "auction_month") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = normalize_text(df[col])
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() > 0:
        df[col] = numeric
        return df
    key = s.str.slice(0, 3)
    df[col] = key.map(MONTH_MAP)
    return df


def coerce_dow(df: pd.DataFrame, col: str = "auction_dow") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = normalize_text(df[col])
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() > 0:
        df[col] = numeric
        return df
    key = s.str.slice(0, 3)
    df[col] = key.map(DOW_MAP)
    return df


def coerce_accidents(df: pd.DataFrame, col: str = "accidents") -> pd.DataFrame:
    if col not in df.columns:
        return df
    s = normalize_text(df[col])
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() > 0:
        df[col] = numeric
        return df

    yes = {"yes", "y", "true", "t", "1", "accident", "reported"}
    no = {"no", "n", "false", "f", "0", "none", "clean"}
    df[col] = s.map(lambda x: 1 if x in yes else (0 if x in no else None))
    return df


def main():
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = coerce_month(df, "auction_month")
    df = coerce_dow(df, "auction_dow")
    df = coerce_accidents(df, "accidents")

    df = coerce_numeric_like(df, [
        "year", "mileage", "owners", "latitude", "longitude",
        "views", "watchers", "comments",
        "auction_month", "auction_dow",
        TARGET
    ])

    df["submodel"] = df["submodel"].astype(str).replace({"nan": "Other", "None": "Other"}).fillna("Other").str.strip()

    # Impute sparse columns
    df["accidents"] = df["accidents"].fillna(0)
    df["accidents"] = (df["accidents"] >= 1).astype(int)

    owners_median = df["owners"].median()
    if pd.isna(owners_median):
        owners_median = 2
    df["owners"] = df["owners"].fillna(owners_median)
    df["owners"] = df["owners"].clip(lower=1, upper=6).astype(int)

    df["auction_month"] = df["auction_month"].clip(lower=1, upper=12)
    df["auction_dow"] = df["auction_dow"].clip(lower=0, upper=6)

    for c in ["views", "watchers", "comments"]:
        df[c] = df[c].clip(lower=0)

    essential = ["submodel", "year", "mileage", "latitude", "longitude",
                 "views", "watchers", "comments", "auction_month", "auction_dow", TARGET]
    df = df.dropna(subset=essential).copy()

    print(f"Rows after cleaning/imputation: {len(df):,}")
    if len(df) < 200:
        raise ValueError("Too few usable rows after cleaning. Something is still being coerced to NaN broadly.")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # RMSE compatible with older sklearn
    rmse = mean_squared_error(y_test, preds) ** 0.5

    r2 = r2_score(y_test, preds)

    print("\nAvant Garde BaT-aligned model (submodel + owners + engagement + timing)")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R²:   {r2:.3f}")

    joblib.dump(pipe, "xgb_pipeline_ag_bat.joblib")
    print("\nSaved: xgb_pipeline_ag_bat.joblib")


if __name__ == "__main__":
    main()
