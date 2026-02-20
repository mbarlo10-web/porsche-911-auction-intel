# Consistency Report: Three Porsche 911 Pricing Project Folders

**Folders compared:**  
- `CIS508_Porsche911_App` (local app)  
- `Porsche_Streamlit_Test` (Streamlit Cloud / GitHub)  
- `AGC_Test` (validation script)

---

## 1. Model & Loading

| Item | CIS508_Porsche911_App | Porsche_Streamlit_Test | AGC_Test |
|------|------------------------|------------------------|----------|
| **Model format** | MLflow (`xgb_price_model_best/`) | joblib (`xgb_pipeline_ag_bat.joblib`) | MLflow (`xgb_price_model_best/`) |
| **Load method** | `mlflow.pyfunc.load_model(MODEL_URI)` | `joblib.load(MODEL_PATH)` | `mlflow.pyfunc.load_model(model_uri)` |
| **Model artifact** | Same MLmodel as AGC_Test (same run_id) | Different pipeline (sklearn + OneHot + XGB) | Same as CIS508 |

**Finding:** CIS508 and AGC_Test use the **same** MLflow model. Porsche_Streamlit_Test uses a **different** model (joblib pipeline from `train_xgb_ag_bat_model.py`).

---

## 2. Feature Sets (What the model receives)

| Feature | CIS508 | Streamlit_Test | AGC_Test |
|---------|--------|----------------|----------|
| **submodel** | ❌ Not passed (UI only) | ✅ Used (OneHot) | ❌ Not in FEATURE_COLS |
| **year** | ✅ | ✅ | ✅ |
| **mileage** | ✅ (+ mileage_corrected, mileage_numeric, mileage_from_title, mileage_from_details) | ✅ | ✅ (+ same mileage_* cols) |
| **owners** | ❌ Not collected | ✅ | ❌ Not in FEATURE_COLS |
| **accidents** | ✅ | ✅ | ✅ |
| **latitude, longitude** | ✅ | ✅ | ✅ |
| **views, watchers, comments** | ✅ (medians from CSV) | ✅ (user sliders) | ✅ (filled with 0 for AGC) |
| **auction_month, auction_dow** | ✅ (fixed best: 3, 4) | ✅ (dynamic / now) | ✅ (filled with 0 for AGC) |

**Finding:**  
- **CIS508 & AGC_Test:** Same 15 numeric features; **no submodel, no owners**.  
- **Porsche_Streamlit_Test:** 12 numeric + **submodel** (categorical); matches `train_xgb_ag_bat_model.py`.

**Important:** CIS508 UI shows submodel and says “Submodel is the single largest driver,” but the prediction **does not use submodel**. Copy and behavior are inconsistent.

---

## 3. Submodel Lists

| | CIS508 | Streamlit_Test |
|--|--------|----------------|
| **Source** | Hardcoded in `st.selectbox` | `load_submodels()` returns hardcoded list |
| **Options** | Same 46 options (Base, Carrera, … 992 GT3, Other) | Same 46 options |
| **Default** | index=7 → "GT3" | "GT3" (with index lookup) |

**Finding:** Submodel **options** are consistent. Only Streamlit_Test actually **uses** submodel in the model.

---

## 4. Data Files

| File | CIS508 | Streamlit_Test | AGC_Test |
|------|--------|----------------|----------|
| **BaT CSV** | `final_predictions_clean_with_corrected_mileage.csv` | Same | Not used |
| **AGC CSV** | Not used | Not used | `AGC_Porsche_Sales_2023-2025.csv` |
| **zip_code lookup** | From BaT CSV | From BaT CSV | STORE_INFO (AG1→97227, AG2→85255) |

**Finding:** BaT CSV path/usage consistent where applicable. AGC_Test correctly uses its own CSV and store→zip mapping.

---

## 5. Timing Logic

| | CIS508 | Streamlit_Test |
|--|--------|----------------|
| **Type** | Static | Dynamic (90-day scan) |
| **Month** | BEST_MONTH = 3 → “February to April” | Uses current month for “today,” then scans next 90 days |
| **Day of week** | BEST_DOW = 4 → “Thursday or Friday” | Prefers Thu/Fri if within 1% of best |
| **Output** | Single recommendation | Best date + top 5 dates table |

**Finding:** Timing behavior differs by design; no bug, but worth being aware for slides/docs.

---

## 6. UI / Inputs

| Input | CIS508 | Streamlit_Test |
|-------|--------|----------------|
| Year | ✅ | ✅ |
| Mileage | ✅ | ✅ |
| Submodel | ✅ (not used in model) | ✅ (used in model) |
| Auction Title | ✅ (not used) | ❌ |
| ZIP Code | ✅ | ✅ |
| Owners | ❌ | ✅ |
| Views / Watchers / Comments | ❌ (medians only) | ✅ Sliders + presets |
| Accidents | ❌ (median only) | ✅ Radio Yes/No |

**Finding:** Streamlit_Test exposes more of the model’s features; CIS508 has extra UI (title, submodel) that doesn’t affect the prediction.

---

## 7. Dependencies

| | CIS508 | Porsche_Streamlit_Test | AGC_Test |
|--|--------|------------------------|----------|
| **requirements.txt** | streamlit, mlflow, pandas, numpy, scikit-learn, xgboost | None at repo root | Not present |
| **Model load** | mlflow | joblib | mlflow |

**Finding:** CIS508 is the only one with an explicit app-level requirements file. Streamlit_Test and AGC_Test rely on implicit/env deps.

---

## 8. Summary of Inconsistencies

1. **Two different models**
   - CIS508 & AGC_Test: MLflow model, **no submodel, no owners** in features.
   - Porsche_Streamlit_Test: joblib pipeline, **submodel + owners** in features.

2. **CIS508 submodel**
   - Submodel is shown and described as a key driver but **not sent to the model**. Either add submodel to the MLflow feature set and retrain, or change the copy (e.g. “Submodel not yet used in this model”).

3. **AGC_Test vs deployed model**
   - AGC_Test uses the **CIS508/MLflow** feature set (no submodel). It does **not** use the Streamlit_Test/joblib model. Validation numbers you quoted for “the model” are for the MLflow model on AGC 911 data.

4. **Submodel list**
   - Same 46 options in both apps; only Streamlit_Test uses them in the model.

5. **Docs/deps**
   - Only CIS508 has a `requirements.txt`; Streamlit_Test and AGC_Test do not list dependencies at repo level.

---

## 9. Recommendations

1. **Clarify in READMEs/slides** which app uses which model (MLflow vs joblib) and that AGC validation is for the MLflow model.
2. **CIS508:** Either (a) use the joblib pipeline and add submodel/owners to the app, or (b) keep MLflow and update UI/copy so it doesn’t claim submodel drives the prediction.
3. **AGC_Test:** If you want to validate the **Streamlit_Test** model, add a path that loads `xgb_pipeline_ag_bat.joblib`, builds features with submodel + owners (and same preprocessing as `train_xgb_ag_bat_model.py`), and runs predictions on AGC 911 data.
4. **Requirements:** Add a `requirements.txt` to Porsche_Streamlit_Test and AGC_Test (and optionally a short README in AGC_Test) so all three folders are consistent and runnable.

If you tell me which direction you prefer (e.g. “make CIS508 use the joblib model” or “keep two models but document them”), I can outline exact code changes next.
