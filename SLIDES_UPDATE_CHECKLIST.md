# Slides Update Checklist — Avant Garde 12:30 Presentation

Use this with your deck so everything matches the **Porsche_Streamlit_Test** app and the “accurate price → reserve met” goal.

---

## 1. App / demo slide

- [ ] **URL:** Use the live Streamlit Cloud URL from your deploy (e.g. `https://porsche-911-auction-intel-xxxxx.streamlit.app`). If you only updated the same repo, the old URL may still work—confirm it opens the XGBoost app with timing recommender.
- [ ] **One-liner:** “Recommended app for pricing and timing: Porsche 911 Auction Intelligence (Streamlit); goal is accurate price to avoid reserve not met.”

---

## 2. Model / performance slide

- [ ] **Model name:** Say **XGBoost** (selected after evaluating LightGBM, k-NN, neural net).
- [ ] **Metrics (BaT test set):**
  - R² = **0.823** (82.3% of price variance explained)
  - Mean Absolute Error = **$18,825**
  - Mean Absolute % Error = **20.3%**
  - Test set: **934** auctions
- [ ] **Training:** ~**4,700** BaT auctions.
- [ ] **Message:** “Model is production-ready for BaT-style auction pricing and supports reserve-setting decisions.”

---

## 3. Validation / AGC slide

- [ ] **Sample size:** “**216** AGC Porsche 911 sales” (not 302).
- [ ] **Best cases:** “Best predictions when AGC data aligns with BaT features: within **$36–$1,277** of actual.”
- [ ] **Overall AGC:** “Overall error on AGC is higher (mean error ~$55K, 36% MAPE) because AGC data is basic—no engagement metrics, listing details, or auction timing.”
- [ ] **Takeaway:** “When features align, accuracy is strong; improving AGC data completeness would improve validation and support dealership-grade valuation.”

---

## 4. Ultimate goal / value slide

- [ ] **Goal:** “**Ultimate goal:** Help determine the most accurate price so auction reserve prices are met—reducing costly ‘reserve not met’ outcomes.”
- [ ] **Role of app:** “The Streamlit app is the recommended tool for pricing and timing so sellers can set data-driven reserves and list at times that maximize the chance of a successful sale.”

---

## 5. Optional (if time)

- [ ] **AGC stats (all Porsche):** If you show AGC stats, use: **307** Porsche sales; AG1 **163**, AG2 **144**; peak volume month **July** (40); highest avg price month **April** ($217,447); peak day **Friday** (74); highest price day **Wednesday** ($149,918).
- [ ] **Two models:** “We use the XGBoost app (Porsche_Streamlit_Test) for daily pricing and timing; AGC Test is a standalone validation project for the less detailed AGC dataset.”

---

## One-line summary for closing

**“We use the Streamlit XGBoost app to get the most accurate price and best close dates so reserves are set to be met and ‘reserve not met’ outcomes are reduced.”**
