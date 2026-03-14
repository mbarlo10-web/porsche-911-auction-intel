# Porsche 911 Auction Intelligence (Streamlit)

**Ultimate goal:** Help determine the most accurate price so auction reserve prices are met—reducing "reserve not met" outcomes on Bring-a-Trailer–style auctions.

ML-powered price advisor for Porsche 911 auctions. Uses an XGBoost model trained on BaT data, now enhanced with simple NLP features from the listing description plus a 90-day timing recommender.

This repository is the **V2, NLP-enhanced** version of the project.  
The original **structured-only V1** implementation lives in a separate repo (`porsche-auction-intel`) and is kept for history/comparison.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set **Main file path** to `streamlit_app.py`.
4. Deploy (no secrets required for basic run).

## Contents

- `streamlit_app_dev.py` — main app logic (dealer UI + NLP + top 5 close dates)
- `streamlit_app.py` — thin entrypoint that calls `main()` from `streamlit_app_dev.py`
- `train_xgb_ag_bat_model.py` — training script for the baseline model
- `train_xgb_ag_bat_model_nlp_compare.py` — baseline vs NLP comparison training
- `xgb_pipeline_ag_bat.joblib` — baseline XGBoost pipeline (used for timing module)
- `xgb_pipeline_ag_bat_nlp_compare.joblib` — NLP-enhanced XGBoost pipeline (used for price estimates)
- `final_predictions_clean_with_corrected_mileage.csv` — BaT-derived dataset used for stats/zip lookup
- `BAT_MODEL_PERFORMANCE_EXPLANATION.md` — performance summary for stakeholders
