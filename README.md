# Porsche 911 Auction Intelligence (Streamlit)

**Ultimate goal:** Help determine the most accurate price so auction reserve prices are met—reducing "reserve not met" outcomes on Bring-a-Trailer–style auctions.

ML-powered price advisor for Porsche 911 auctions. Uses XGBoost trained on BaT data; supports pricing and 90-day timing recommendations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app_dev.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set **Main file path** to `streamlit_app_dev.py`.
4. Deploy (no secrets required for basic run).

## Contents

- `streamlit_app_dev.py` — main app (price prediction + top 5 close dates)
- `train_xgb_ag_bat_model.py` — training script for the joblib model
- `xgb_pipeline_ag_bat.joblib` — trained model (must be in repo for Cloud)
- `final_predictions_clean_with_corrected_mileage.csv` — used for stats/zip lookup
- `BAT_MODEL_PERFORMANCE_EXPLANATION.md` — performance summary for stakeholders
