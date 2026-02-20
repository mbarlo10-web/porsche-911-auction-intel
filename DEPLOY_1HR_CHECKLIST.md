# 1-Hour Deploy Checklist — Porsche_Streamlit_Test to GitHub + Streamlit

**Goal:** Get the app on GitHub and Streamlit Cloud before your 12:30 Avant Garde presentation.

---

## Pre-flight (≈5 min)

- [ ] **Confirm large files:** Repo has `final_predictions_clean_with_corrected_mileage.csv` (~4.8 MB) and `xgb_pipeline_ag_bat.joblib` (~4.4 MB). GitHub allows files < 100 MB; both are fine.
- [ ] **Confirm assets:** `avant_garde_logo.png`, `GT3.jpg`, `asu-wpcarey-school-of-business-asu-footer.png` are in the folder (required for app UI).

---

## Step 1: Initialize Git and push to GitHub (≈15 min)

1. **Create a new repo on GitHub**
   - GitHub.com → New repository
   - Name (e.g.) `porsche-911-auction-intel` or `Porsche_Streamlit_Test`
   - Public, no README/license/ignore (you already have files)

2. **In terminal, from `Porsche_Streamlit_Test` folder:**
   ```bash
   cd /Users/markbarlow/Desktop/Porsche_Streamlit_Test

   git init
   git add .
   git commit -m "Porsche 911 Auction Intelligence - XGBoost app + timing recommender"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub username and repo name.

3. **If you already have a repo** that you want to replace with this app:
   - Clone it, replace contents with `Porsche_Streamlit_Test` (keep `.git`), then add/commit/push.

---

## Step 2: Deploy on Streamlit Cloud (≈10 min)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

2. **New app**
   - Click "New app"
   - **Repository:** select the repo you just pushed (e.g. `YOUR_USERNAME/porsche-911-auction-intel`)
   - **Branch:** `main`
   - **Main file path:** `streamlit_app_dev.py`  ← must be exactly this
   - **App URL:** choose a slug (e.g. `porsche-911-auction-intel`)

3. Click **Deploy**. First run can take 3–5 minutes (installs deps, loads model).

4. **If deploy fails**
   - Check build logs for missing packages; add them to `requirements.txt` and push again.
   - Ensure `xgb_pipeline_ag_bat.joblib` and `final_predictions_clean_with_corrected_mileage.csv` are in the repo (not in `.gitignore`).

5. **Copy the live URL** (e.g. `https://porsche-911-auction-intel-xxxxx.streamlit.app`) for your slides.

---

## Step 3: Update slides (≈20 min)

Use **SLIDES_UPDATE_CHECKLIST.md** in this folder for a concise list of which slides to change and what to say.

**Must-do for 12:30:**
- **App demo slide:** Use the **new** Streamlit Cloud URL from Step 2 (or keep existing if you’re only updating the same app’s repo).
- **Model slide:** Say “XGBoost” (selected over LightGBM); use metrics: **82.3% R², ~$18,825 MAE, 20.3% MAPE** (BaT test set).
- **Validation slide:** “216 AGC Porsche 911 sales”; “best predictions within $36–$1,277 when features align”; “overall AGC error higher due to basic dataset.”
- **Goal:** “Ultimate goal: most accurate price to avoid reserve not met.”

---

## Step 4: Buffer (≈10 min)

- Open the live app in a browser tab before the meeting.
- Do one test run: enter a car, click “Estimate Auction Price,” confirm price and top 5 dates appear.
- Have a backup: “If the live app is slow, the model runs locally and we can demo from here.”

---

## Quick reference

| Item | Value |
|------|--------|
| Main app file | `streamlit_app_dev.py` |
| Model file | `xgb_pipeline_ag_bat.joblib` |
| BaT test metrics | R² 0.823, MAE $18,825, MAPE 20.3% |
| AGC validation | 216 Porsche 911 sales; best cases ~$36–$1,277 error |
| Ultimate goal | Accurate price → reserve met |

Good luck with the presentation.
