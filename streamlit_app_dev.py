import os
import datetime as dt

import joblib
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

# Optional timezone support
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

# Baseline model is still used for timing recommendations.
MODEL_PATH = os.path.join(BASE_DIR, "xgb_pipeline_ag_bat.joblib")

# Primary pricing model for the dealer-facing dashboard.
NLP_MODEL_PATH = os.path.join(BASE_DIR, "xgb_pipeline_ag_bat_nlp_compare.joblib")
DATA_PATH = os.path.join(BASE_DIR, "final_predictions_clean_with_corrected_mileage.csv")

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Timing recommender controls
HORIZON_DAYS = 90
PREFERRED_DOWS = {3, 4}        # Thu/Fri
BIAS_TOLERANCE_PCT = 0.01      # prefer Thu/Fri if within 1% of best


# ------------------------------------------------------------------
# TIME HELPERS
# ------------------------------------------------------------------

def _now_phoenix():
    if ZoneInfo is None:
        return dt.datetime.now()
    return dt.datetime.now(ZoneInfo("America/Phoenix"))


def format_date_dMY(d):
    """Day Month Year with month spelled out (e.g., 02 March 2026)."""
    return d.strftime("%d %B %Y")


# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_nlp_model():
    """Load the experimental NLP-enhanced model if available."""
    if not os.path.exists(NLP_MODEL_PATH):
        return None
    return joblib.load(NLP_MODEL_PATH)

@st.cache_data
def load_training_stats(csv_path):
    df = pd.read_csv(csv_path)

    med_lat = float(df["latitude"].median())
    med_lon = float(df["longitude"].median())

    med_views = int(df["views"].median()) if "views" in df.columns else 8000
    med_watchers = int(df["watchers"].median()) if "watchers" in df.columns else 120
    med_comments = int(df["comments"].median()) if "comments" in df.columns else 25

    # owners and accidents are sparse in your dataset. Provide sensible defaults.
    if "owners" in df.columns and is_numeric_dtype(df["owners"]):
        med_owners = int(round(df["owners"].dropna().median())) if df["owners"].dropna().shape[0] > 0 else 2
    else:
        med_owners = 2

    if "accidents" in df.columns and is_numeric_dtype(df["accidents"]):
        med_accidents = int(round(df["accidents"].dropna().median())) if df["accidents"].dropna().shape[0] > 0 else 0
    else:
        med_accidents = 0

    # Slider bounds (10th-90th percentile)
    def bounds(col, fallback_lo, fallback_hi):
        if col in df.columns and df[col].dropna().shape[0] > 0:
            lo = int(df[col].quantile(0.10))
            hi = int(df[col].quantile(0.90))
            return lo, max(hi, lo + 1)
        return fallback_lo, fallback_hi

    views_lo, views_hi = bounds("views", 0, 25000)
    watchers_lo, watchers_hi = bounds("watchers", 0, 1500)
    comments_lo, comments_hi = bounds("comments", 0, 150)

    return (
        med_lat, med_lon,
        med_views, med_watchers, med_comments,
        views_lo, views_hi,
        watchers_lo, watchers_hi,
        comments_lo, comments_hi,
        med_owners, med_accidents,
    )


@st.cache_data
def load_zip_lookup(csv_path):
    df = pd.read_csv(csv_path, usecols=["zip_code", "latitude", "longitude"])
    df = df.dropna(subset=["zip_code", "latitude", "longitude"]).drop_duplicates(subset=["zip_code"])
    df["zip_str"] = df["zip_code"].astype(float).astype(int).astype(str)
    return df[["zip_str", "latitude", "longitude"]]


def get_lat_lon_for_zip(zip_df, zipcode_str, default_lat, default_lon):
    if zipcode_str is None:
        return default_lat, default_lon

    z_str = zipcode_str.strip()
    if not z_str:
        return default_lat, default_lon

    match = zip_df[zip_df["zip_str"] == z_str]
    if match.empty:
        return default_lat, default_lon

    lat = float(match["latitude"].iloc[0])
    lon = float(match["longitude"].iloc[0])
    return lat, lon


@st.cache_data
def load_submodels(csv_path):
    # Comprehensive list of Porsche 911 submodels
    return [
        "Base",
        "Carrera",
        "Carrera S",
        "Carrera 4",
        "Carrera 4S",
        "Carrera GTS",
        "Carrera 4 GTS",
        "Targa",
        "Targa 4",
        "Targa 4S",
        "Targa GTS",
        "Turbo",
        "Turbo S",
        "GT3",
        "GT3 RS",
        "GT3 Touring",
        "GT2",
        "GT2 RS",
        "Speedster",
        "Safari",
        "SC",
        "RS",
        "RS America",
        "Carrera 3.2",
        "Longhood",
        "930 Turbo",
        "964",
        "964 Turbo",
        "993",
        "993 Turbo",
        "993 GT2",
        "996",
        "996 Turbo",
        "996 GT2",
        "996 GT3",
        "997",
        "997 Turbo",
        "997 GT2",
        "997 GT3",
        "991",
        "991 Turbo",
        "991 GT2",
        "991 GT3",
        "992",
        "992 Turbo",
        "992 GT3",
        "Other",
    ]


# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------

def set_page_style():
    st.set_page_config(
        page_title="Porsche 911 Auction Intelligence",
        page_icon="🚗",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            body { background-color: #000; color: #f9f9f9; }
            .main { background-color: #000; color: #f9f9f9; }
            h1, h2, h3, h4 { color: #f7941d; }

            /* Make all regular text brighter and slightly bolder for readability */
            p, li, span, label, .stText, .stMarkdown, .stRadio, .stSelectbox, .stSlider {
                color: #f2f2f2 !important;
                font-weight: 500;
            }

            /* Research expander styling */
            div[data-testid="stExpander"] {
                border-radius: 12px;
                border: 1px solid #f7941d;
            }
            div[data-testid="stExpander"] div[data-testid="stExpanderHeader"] p {
                color: #f7941d !important;
                font-size: 1.0rem;
                font-weight: 700;
            }

            .avant-card {
                background-color: #111;
                border-radius: 12px;
                padding: 1.2rem 1.4rem;
                border: 1px solid #333;
            }
            .avant-metric {
                font-size: 1.8rem;
                font-weight: 800;
                color: #f7941d;
            }
            .avant-subtitle {
                color: #f5f5f5;
                font-size: 0.98rem;
                font-weight: 600;
            }
            .avant-small {
                color: #f0f0f0;
                font-size: 0.96rem;
                line-height: 1.45;
                font-weight: 500;
            }

            .stButton>button {
                background-color: #f7941d; color: black;
                border-radius: 999px; padding: 0.5rem 1.4rem;
                font-weight: 600;
            }
            .stButton>button:hover { background-color: #ffae42; }

            /* Brand links under logo */
            .brand-links {
                margin-top: 0.4rem;
                margin-bottom: 0.8rem;
                line-height: 1.4;
            }
            .brand-links a {
                color: #f7941d;
                text-decoration: none;
                font-weight: 600;
                font-size: 0.95rem;
            }
            .brand-links a:hover {
                text-decoration: underline;
                color: #ffae42;
            }
            .brand-links span {
                color: #777;
                padding: 0 0.35rem;
            }

            /* ---------------------------------------------------------
               Option A: Highlight DataFrame with Avant Garde orange theme
               --------------------------------------------------------- */
            div[data-testid="stDataFrame"] {
                background-color: #1a1208;
                border-radius: 12px;
                padding: 0.6rem;
                border: 1px solid #f7941d;
            }
            div[data-testid="stDataFrame"] thead th {
                background-color: #f7941d !important;
                color: #000 !important;
                font-weight: 700 !important;
            }
            div[data-testid="stDataFrame"] tbody td {
                background-color: #24170a;
                color: #fff;
            }
            div[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
                background-color: #2a1c0d;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# TIMING RECOMMENDER
# ------------------------------------------------------------------

def recommend_best_timing(row_base: dict, model, horizon_days=HORIZON_DAYS, tolerance_pct=BIAS_TOLERANCE_PCT):
    start = _now_phoenix()
    dates = [start + dt.timedelta(days=i) for i in range(horizon_days + 1)]

    rows = []
    for d in dates:
        r = row_base.copy()
        r["auction_month"] = int(d.month)
        r["auction_dow"] = int(d.weekday())
        r["_date"] = d.date()
        rows.append(r)

    cand = pd.DataFrame(rows)
    cand["pred_price"] = model.predict(cand.drop(columns=["_date"], errors="ignore"))

    best_abs = cand.loc[cand["pred_price"].idxmax()].copy()
    best_abs_price = float(best_abs["pred_price"])

    preferred = cand[cand["auction_dow"].isin(PREFERRED_DOWS)].copy()
    chosen = best_abs

    if not preferred.empty:
        best_pref = preferred.loc[preferred["pred_price"].idxmax()].copy()
        best_pref_price = float(best_pref["pred_price"])
        if best_pref_price >= best_abs_price * (1 - tolerance_pct):
            chosen = best_pref

    top5 = cand.sort_values("pred_price", ascending=False).head(5).copy()
    top5["Day"] = top5["auction_dow"].astype(int).map(lambda x: DOW_NAMES[x])
    top5["Date"] = top5["_date"].apply(format_date_dMY)

    # Clean display formatting (no decimals)
    top5["Predicted Price"] = top5["pred_price"].apply(lambda x: f"${x:,.0f}")

    top5 = top5[["Date", "Day", "Predicted Price"]]

    return chosen, top5


def _compute_listing_text_features(listing_text: str, mileage_value: int) -> dict:
    """
    Derive simple keyword and domain features from free-text listing details.

    Mirrors the training-time features used in the NLP-enhanced comparison model.
    """
    text = (listing_text or "").lower()
    # Normalize whitespace similar to training cleanup
    text = " ".join(text.split())

    manual_transmission = int("manual" in text)

    turbo_engine = int(
        ("turbocharged" in text)
        or ("twin-turbo" in text)
        or ("turbo engine" in text)
    )

    limited_slip_diff = int("limited-slip" in text)
    service_records = int("service records" in text)
    clean_carfax = int("clean carfax" in text)
    sport_package = int("sport chrono" in text)

    ceramic_brakes = int(
        ("ceramic brakes" in text)
        or ("pccb" in text)
        or ("carbon-ceramic" in text)
    )

    pdk_transmission = int("pdk" in text)
    bucket_seats = int(
        ("bucket seats" in text) or ("full bucket seats" in text)
    )

    color_guards_red = int("guards red" in text)
    color_grand_prix_white = int("grand prix white" in text)
    color_speed_yellow = int("speed yellow" in text)

    rare_color = int(
        color_guards_red or color_grand_prix_white or color_speed_yellow
    )

    low_mileage = int(mileage_value is not None and mileage_value < 30000)

    return {
        "manual_transmission": manual_transmission,
        "turbo_engine": turbo_engine,
        "limited_slip_diff": limited_slip_diff,
        "service_records": service_records,
        "clean_carfax": clean_carfax,
        "sport_package": sport_package,
        "ceramic_brakes": ceramic_brakes,
        "pdk_transmission": pdk_transmission,
        "bucket_seats": bucket_seats,
        "color_guards_red": color_guards_red,
        "color_grand_prix_white": color_grand_prix_white,
        "color_speed_yellow": color_speed_yellow,
        "rare_color": rare_color,
        "low_mileage": low_mileage,
    }


# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------

def main():
    set_page_style()

    # Baseline model: internal use for timing.
    model = load_model()
    # NLP-enhanced model: primary pricing model for dealers.
    nlp_model = load_nlp_model()

    (
        med_lat, med_lon,
        med_views, med_watchers, med_comments,
        views_lo, views_hi,
        watchers_lo, watchers_hi,
        comments_lo, comments_hi,
        med_owners, med_accidents,
    ) = load_training_stats(DATA_PATH)

    zip_df = load_zip_lookup(DATA_PATH)
    submodel_options = load_submodels(DATA_PATH)

    # ---------------- HERO ----------------
    banner_left_col, banner_right_col = st.columns([1.2, 4])

    with banner_left_col:
        st.image("avant_garde_logo.png", width=350)

        st.markdown(
            """
            <div class="brand-links">
              <a href="https://a-gc.com" target="_blank" rel="noopener noreferrer">Avant-Garde Collection</a>
              <span>•</span>
              <a href="https://wpcarey.asu.edu/" target="_blank" rel="noopener noreferrer">ASU W. P. Carey School of Business</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h1>Porsche 911 Auction Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<div class='avant-subtitle'>BaT-aligned price advisor for Avant Garde</div>", unsafe_allow_html=True)
        st.markdown("<div class='avant-subtitle'>Mark Barlow • MS AIB Candidate</div>", unsafe_allow_html=True)

    with banner_right_col:
        st.image("GT3.jpg", use_column_width=True)

    st.markdown("---")

    input_col, explain_col = st.columns([1.4, 1])

    # ---------------- INPUTS (LEFT COLUMN) ----------------
    with input_col:
        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.markdown("#### Configure the car")

        years = list(range(1965, 2024))
        year = st.selectbox("Model Year", years, index=years.index(2015))

        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=250000,
            value=30000,
            step=500,
        )

        default_submodel = "GT3"
        if default_submodel in submodel_options:
            default_idx = submodel_options.index(default_submodel)
        else:
            default_idx = 0

        submodel = st.selectbox(
            "Submodel",
            submodel_options,
            index=default_idx,
        )

        owners = st.number_input(
            "Number of owners",
            min_value=1,
            max_value=6,
            value=int(med_owners),
            step=1,
        )

        zipcode = st.text_input("Seller ZIP Code", "85260")

        listing_details_text = st.text_area(
            "Listing details (optional)",
            help=(
                "Paste the auction or dealer description to enable the "
                "experimental NLP-enhanced estimate."
            ),
            height=140,
        )

        st.markdown("#### Expected engagement (demand signal)")

        preset = st.selectbox("Engagement preset", ["Average", "Low", "High"], index=2)

        if preset == "Low":
            views_default = max(views_lo, int(med_views * 0.7))
            watchers_default = max(watchers_lo, int(med_watchers * 0.7))
            comments_default = max(comments_lo, int(med_comments * 0.7))
        elif preset == "High":
            views_default = min(views_hi, int(med_views * 1.3))
            watchers_default = min(watchers_hi, int(med_watchers * 1.3))
            comments_default = min(comments_hi, int(med_comments * 1.3))
        else:
            views_default, watchers_default, comments_default = med_views, med_watchers, med_comments

        views = st.slider("Views", min_value=max(0, views_lo), max_value=views_hi, value=views_default)
        watchers = st.slider("Watchers", min_value=max(0, watchers_lo), max_value=watchers_hi, value=watchers_default)
        comments = st.slider("Comments", min_value=max(0, comments_lo), max_value=comments_hi, value=comments_default)

        st.markdown("#### Accidents")
        st.markdown(
            "<div class='avant-small'>Select whether accidents are reported. Accident history can reduce buyer confidence and pricing.</div>",
            unsafe_allow_html=True,
        )
        accident_choice = st.radio("Accidents reported?", ["No", "Yes"], horizontal=True, index=0)
        accidents = 1 if accident_choice == "Yes" else 0

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- EXPLANATION (RIGHT COLUMN) ----------------
    with explain_col:
        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.subheader("What drives price in this model")
        st.markdown(
            """
- **Submodel** captures trim hierarchy and performance demand (Turbo, GT3, GT3 RS).
- **Owners** and accident history reflect provenance and perceived risk.
- **Year** and mileage capture generation and condition effects.
- **Location** (lat/long) reflects regional demand and buyer density.
- **Engagement** (views, watchers, comments) acts as a demand signal.
- **Timing** is used internally to recommend a strong close date.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Move this block down to align with "Expected engagement"
        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)

        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.subheader("Why it matters")
        st.markdown(
            """
Views, watchers, and comments proxy buyer attention. More attention often increases bidding competition, which can improve the final sold price.

Use **Average** as a baseline. Select **Low** when demand feels softer or seasonally constrained. Select **High** when listings are expected to receive strong exposure or cross-promotion.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- RESULTS ----------------
    st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
    st.markdown("### Price Estimate")

    if st.button("Estimate Auction Price"):
        lat, lon = get_lat_lon_for_zip(zip_df, zipcode, med_lat, med_lon)

        now = _now_phoenix()

        base_row = {
            "submodel": submodel,
            "year": int(year),
            "mileage": int(mileage),
            "owners": int(owners),
            "accidents": int(accidents),
            "latitude": float(lat),
            "longitude": float(lon),
            "views": int(views),
            "watchers": int(watchers),
            "comments": int(comments),
            "auction_month": int(now.month),
            "auction_dow": int(now.weekday()),
        }

        row_today = pd.DataFrame([base_row])

        # Price estimate from NLP-enhanced model when available.
        nlp_features = _compute_listing_text_features(
            listing_details_text, int(mileage)
        )
        row_for_price = pd.DataFrame([{**base_row, **nlp_features}])

        if nlp_model is not None:
            predicted_price_today = float(nlp_model.predict(row_for_price)[0])
        else:
            # Fallback to baseline if NLP model artifact is missing.
            predicted_price_today = float(model.predict(row_today)[0])

        best, top5 = recommend_best_timing(base_row, model)
        best_date_str = format_date_dMY(best["_date"])
        best_dow = DOW_NAMES[int(best["auction_dow"])]

        st.markdown("#### Estimated sold price")

        c1, c2, c3 = st.columns([1.3, 1, 1])

        with c1:
            st.markdown("Predicted sold price (if listed today)")
            st.markdown(f"<div class='avant-metric'>${predicted_price_today:,.0f}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("Recommended close date")
            st.markdown(f"<div class='avant-metric'>{best_date_str}</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("Recommended day")
            st.markdown(f"<div class='avant-metric'>{best_dow}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"""
This estimate assumes a **{year} {submodel}** with **{mileage:,} miles**, **{owners} owner(s)**, listed from ZIP **{zipcode}**.  
Expected engagement: **{views:,} views**, **{watchers:,} watchers**, **{comments:,} comments**.  
Accidents reported: **{accident_choice}**.
            """
        )

        if listing_details_text and listing_details_text.strip():
            st.markdown(
                "<div class='avant-small'>"
                "This estimate incorporates text-derived signals from the "
                "listing details (manual gearbox, Turbo/PCCB, colors, "
                "service records, and more)."
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"#### Top 5 predicted close dates (next {HORIZON_DAYS} days)")
        top5 = top5.reset_index(drop=True)
        st.dataframe(top5, use_container_width=True, hide_index=True)

        st.markdown("#### How timing is evaluated")
        st.markdown(
            "The model simulates close dates across the next 90 days and estimates the expected sold price for each date. "
            "If a non-Thursday or Friday close is projected to perform better, it will be selected."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RESEARCH & MODEL INSIGHTS (HIDDEN BY DEFAULT) ----------------
    with st.expander("Research & model insights (for analysts and technical readers)", expanded=False):
        st.markdown("<div class='avant-card'>", unsafe_allow_html=True)
        st.markdown("### 1. Model overview", unsafe_allow_html=False)
        st.markdown(
            """
- **Target**: sold price on BaT-style Porsche 911 auctions (future work: Avant-Garde sales).
- **Structured inputs**: year, mileage, submodel, number of owners, accidents, ZIP / latitude / longitude, views, watchers, comments, timing.
- **Text‑derived inputs**: gearbox (manual vs PDK), Turbo keywords, PCCB / ceramic brakes, Sport Chrono, bucket seats, rare colors (Guards Red, Grand Prix White, Speed Yellow), low mileage, service records, clean Carfax, and a combined rare‑color flag.
- **Takeaway**: combining **structured + listing text** outperforms structured data alone on holdout auctions.
            """
        )

        perf_data = pd.DataFrame(
            [
                {
                    "Model": "Baseline (structured only)",
                    "RMSE": 34361,
                    "MAE": 18825,
                    "R2": 0.823,
                },
                {
                    "Model": "NLP-enhanced (with listing text)",
                    "RMSE": 32319,
                    "MAE": 17940,
                    "R2": 0.844,
                },
            ]
        )

        st.markdown("### 2. Performance comparison (baseline vs NLP)")
        st.markdown("#### Holdout performance (BaT data)")
        st.dataframe(perf_data, use_container_width=True, hide_index=True)

        st.markdown("#### RMSE comparison")
        rmse_df = perf_data[["Model", "RMSE"]].set_index("Model")
        st.bar_chart(rmse_df)

        st.markdown(
            "- **RMSE improvement**: NLP reduces RMSE by ~6% and MAE by ~5% on the 20% holdout set.\n"
            "- **Interpretation**: listing descriptions provide incremental signal beyond year / mileage / engagement."
        )

        st.markdown("### 3. Feature importance & domain insights")

        if nlp_model is not None:
            try:
                prep = nlp_model.named_steps["prep"]
                feature_names = prep.get_feature_names_out()
                importances = nlp_model.named_steps["model"].feature_importances_
                fi = (
                    pd.Series(importances, index=feature_names)
                    .sort_values(ascending=False)
                    .head(20)
                )

                st.markdown("#### Top features in the NLP-enhanced model")
                st.bar_chart(fi.sort_values(ascending=True))

                st.markdown(
                    """
- **Low mileage & rare colors**: low mileage, Guards Red, Grand Prix White, and Speed Yellow show strong positive influence.
- **Text signals**: Turbo / GT cars with PCCB, Sport Chrono, and bucket seats command significant premiums.
- **Engagement vs spec**: listing spec and provenance features (gearbox, brakes, color, service history) complement engagement (views / watchers / comments).
                    """
                )
            except Exception:
                st.markdown(
                    "<div class='avant-small'>Unable to display feature importance "
                    "for the current NLP model artifact.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='avant-small'>NLP-enhanced model artifact not found; "
                "research charts are based on the last recorded metrics.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 5. Data & methodology notes")
        st.markdown(
            """
- **Data source**: Bring a Trailer Porsche 911 auction dataset (Kaggle export), filtered to clean sales with usable pricing and mileage.
- **Split**: 80/20 train/holdout with a fixed random seed for reproducibility.
- **Model**: XGBoost regressor wrapped in a scikit-learn pipeline with one‑hot encoding for `submodel` and passthrough numeric features.
- **Limitations**:
  - BaT buyer base and presentation standards may differ from Avant-Garde.
  - Text features are keyword‑based (not a full language model).
  - No private-sale data or off‑platform transactions are captured.
            """
        )

        st.markdown("### 6. Roadmap / future work")
        st.markdown(
            """
- **Avant-Garde data**: scrape and integrate Avant-Garde Porsche sales to fine‑tune the model on in‑house inventory.
- **Richer text modeling**: move from keyword features to embeddings / transformer‑based text encoders.
- **Calibration & monitoring**: calibrate predictions to realized prices and monitor drift over time as market conditions change.
            """
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- FOOTER ----------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns([1, 4, 1])
    with footer_col2:
        st.image(
            "asu-wpcarey-school-of-business-asu-footer.png",
            width=350,
            caption="Powered by ASU W. P. Carey School of Business Research",
        )


if __name__ == "__main__":
    main()
