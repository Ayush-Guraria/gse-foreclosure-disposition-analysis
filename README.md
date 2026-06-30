# GSE Foreclosure Disposition Analysis

**Predicting whether a foreclosed property will sell at auction (3rd-party sale) or revert to the bank as REO (Real Estate Owned)**

> April 2026 | Solo analytical project

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Who It Is Useful For](#who-it-is-useful-for)
- [The Problem](#the-problem)
- [Data Analyzed](#data-analyzed)
- [External Data Sources](#external-data-sources)
- [Analysis Performed](#analysis-performed)
- [Model Used and Why](#model-used-and-why)
- [Results](#results)
- [How to Interpret the Results](#how-to-interpret-the-results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Deliverables](#deliverables)
- [Caveats](#caveats)
- [Next Steps](#next-steps)

---

## What This Project Does

This project builds a machine learning model that predicts the outcome of a foreclosure auction before it happens. Specifically, it answers:

> *Will this property be purchased by a third-party investor at auction, or will it revert back to the bank?*

A foreclosed property that sells at auction is a clean, fast resolution. One that reverts to bank ownership (called REO — Real Estate Owned) costs the bank an estimated **$50,000 to $80,000** in holding costs, maintenance, taxes, and eventual resale expenses.

I processed roughly **280 GB of raw mortgage data**, built **57 engineered features** by joining it to 30+ external data sources, and trained a LightGBM gradient-boosted model that achieves **ROC-AUC of 0.80** — meaning it correctly identifies which of two cases will sell 80% of the time, compared to 50% for random guessing.

The model produces a probability score for each case at intake. That score gets translated into a decile ranking and a routing action, giving operations teams a clear triage signal before any auction marketing or preparation begins.

---

## Who It Is Useful For

**Foreclosure services companies and mortgage servicers**
Foreclosure services companies manage this process end-to-end — legal steps, auction, and post-sale. A triage score at intake lets their operations teams differentiate case handling by predicted outcome rather than treating every case the same way.

**GSE servicers and investors (Fannie Mae / Freddie Mac ecosystem)**
Anyone managing a portfolio of GSE-backed defaulted loans who wants to understand which properties are likely to self-resolve at auction versus which will need active loss-mitigation work.

**Credit risk and portfolio management teams at banks**
The model and methodology can be adapted to forecast REO rates across portfolios, stress-test exposure to market regime shifts, and plan operational capacity.

**Data scientists working in mortgage / real estate analytics**
The pipeline demonstrates how to join public GSE data with FRED macro series, Zillow home values, Realtor.com inventory data, Census demographics, BLS wages, and IRS migration data — including the non-obvious encoding fixes that make these joins actually work at scale.

---

## The Problem

When a homeowner defaults on a GSE-backed mortgage (one owned by Fannie Mae or Freddie Mac), the bank forecloses and takes the property to a public auction. Two outcomes are possible:

**Outcome A — Third-party sale (the good one)**
An external investor bids on the property and buys it. The bank recovers most of what it is owed, quickly, with minimal ongoing cost. Clean exit.

**Outcome B — REO reversion (the expensive one)**
No investor bids enough. The property reverts back to the bank, which now has to pay property taxes, maintain the building, hire a real estate agent, and wait months to sell it on the open market. All-in cost: roughly **$65,000 per property**.

The challenge is that banks and servicers do not know ahead of time which outcome will happen. They treat every case more or less the same until the auction result comes in. This project asks: can we predict the outcome early enough to actually change what we do about it?

---

## Data Analyzed

### Primary source: GSE loan-performance data

Fannie Mae and Freddie Mac are government-sponsored enterprises that buy mortgages from banks and guarantee them. They publish detailed, anonymized data on every loan they own — from the day it was originated through every monthly payment to whatever happened at the end. This is the gold standard for mortgage research because it covers roughly 50% of all US conforming mortgages, it is publicly available, and it gives you a clean verified label for what happened at disposition.

| Source | Files | Raw size | Liquidation cases |
|---|---|---|---|
| Fannie Mae | 40 quarterly ZIP files | ~240 GB | 12,042 |
| Freddie Mac | 9 yearly ZIP files | ~40 GB | 6,829 |
| **Total** | **49 files** | **~280 GB** | **18,871** |

After ingesting all 280 GB, filtering to foreclosure liquidation events only (Zero Balance Codes 02 and 09), and filtering to the 2016-2024 disposition window, I ended up with **18,871 clean cases** to model on.

The target variable is well-balanced: **51.6% sold** vs **48.4% reverted to REO** — no resampling or class weighting needed.

### Why this much data matters

The raw 280 GB exists because the GSE files contain the full monthly performance history of every loan — sometimes hundreds of rows per loan covering years of payment records. We collapse that down to one row per loan at the point of liquidation, keeping only the fields relevant to disposition prediction. This compression step is what produces the 18,871-row analytical dataset from the 280 GB of raw input.

### Three critical data bugs fixed

The raw data had three issues that would have silently destroyed the analysis:

**Bug 1 — Fannie's LTV column was being misread**
The pipeline detected columns by their content patterns. Fannie's interest rate column (values 2–8%) was being mistaken for the LTV column (values 15–97%). This left the original LTV feature at 36% coverage for Fannie loans. A stricter detection rule fixed it. Coverage went from 36% to 100%.

**Bug 2 — Wrong FRED series ID for mortgage delinquency**
The FRED series ID being used (`DRSFRMACBSQ`) returned empty data. The correct one is `DRSFRMACBS`. One character difference, completely silent failure.

**Bug 3 — ZIP code extraction was producing wrong prefixes**
This was the biggest one. GSE data publishes only 3-digit ZIP prefixes for privacy. But Fannie pads them with leading zeros (`00360` — the real prefix is `360`, the last three characters), while Freddie pads with trailing zeros (`61000` — the real prefix is `610`, the first three characters). The naive `zip[:3]` extraction was producing Puerto Rico prefixes for every Fannie row, causing every neighbourhood data join to silently fail and return nothing. After fixing with source-aware extraction, home value and neighbourhood feature coverage jumped from 36% to 99%.

---

## External Data Sources

Loan-only models reach roughly AUC 0.71. The remaining performance comes from understanding the market the auction is happening in — specifically, what an investor sees when deciding whether to bid. I organized external sources into eight categories aligned with the investor decision framework:

| Category | Key sources | What it captures |
|---|---|---|
| Investor economics | Zillow ZHVI, ZORI, ZHVF | Home values, rents, and forward price expectations at ZIP level |
| Market supply and demand | Realtor.com metro inventory, FRED supply series | Active listings, days on market, months of supply |
| Buyer psychology | UMich consumer sentiment (UMCSENT), inflation expectations (MICH), nonfarm payrolls (PAYEMS) | Buyer confidence and employment trajectory |
| Local economic conditions | FHFA MSA HPI, BLS QCEW wages, IRS county migration | Metro price levels, incomes, population movement |
| Credit and capital markets | ICE BofA high-yield spread, Fed lending standards survey, household debt service ratios | How much capital investors can access and at what cost |
| Distress and vacancy indicators | Rental vacancy, homeowner vacancy, SF mortgage delinquency rate | Shadow inventory and sector-wide stress levels |
| Location risk | FEMA National Risk Index (excluded — IP-blocked during cloud processing) | Natural hazard and social vulnerability by county |
| Neighbourhood indicators | Census ACS at ZIP3 level: demographics, housing tenure, rent burden, construction activity | Who lives there, what the housing stock looks like |

In total: **57 engineered features** across 52 numeric and 5 categorical columns.

---

## Analysis Performed

The project ran five types of analysis beyond the core modelling:

**1. Exploratory data analysis**
Sale rates by state (40pp+ spread between top and bottom states), LTV distributions by outcome, sale rate trends over time by channel, top correlations with the target variable.

**2. Model comparison**
Five model approaches tested: logistic regression, LightGBM, XGBoost, CatBoost, an ensemble of all three, per-channel models, and a time-aware split. All compared on ROC-AUC, precision-recall AUC, and training time.

**3. SHAP feature importance**
Computed Shapley values across 2,000 test-set predictions to understand which features drive each individual prediction — not just overall importance rankings but the direction and magnitude of each feature's effect.

**4. Counterfactual channel analysis**
Flipped every test case from judicial to non-judicial (and vice versa) and re-scored the model. This directly tests whether changing the auction venue changes predicted outcomes.

**5. Subgroup performance**
AUC broken out by state (n >= 100), origination vintage, loan source (Fannie vs Freddie), loan size quintile, and channel proxy — to understand where the model is strong and where it is weaker.

**6. Calibration**
Checked whether the model's probability estimates are trustworthy — does a score of 70% actually correspond to a 70% observed sale rate?

**7. Dollar impact modelling**
Translated model accuracy into projected business value under low, medium, and high intervention effectiveness scenarios.

**8. Lead-lag analysis**
For each macro indicator, found the lag at which it correlates most strongly with monthly aggregate sale rate. Identified six leading indicators that move 6-11 months before disposal outcomes shift.

**9. Online vs offline venue test**
Grouped states by online-platform adoption tier, compared raw sale rates and model residuals across tiers — the most defensible indirect test of the online vs offline question given that GSE data does not record actual auction venue.

---

## Model Used and Why

### Selected model: LightGBM with default hyperparameters

**What LightGBM is**
LightGBM is a gradient-boosted decision tree library. It works by building hundreds of decision trees sequentially, where each tree learns from the mistakes of the ones before it. It is fast, handles missing data natively, works well with categorical variables, and is the industry standard for tabular classification problems like this one.

**Why not a more complex model**
I ran five different model families including an ensemble and 40-trial Optuna hyperparameter tuning. Every approach converged to roughly the same AUC:

| Model | ROC-AUC |
|---|---|
| Ensemble (LightGBM + XGBoost + CatBoost) | 0.8016 |
| **LightGBM defaults — selected** | **0.8008** |
| LightGBM tuned via Optuna | 0.8003 |
| CatBoost defaults | 0.7972 |
| XGBoost defaults | 0.7959 |
| Per-channel LightGBM | 0.7947 |
| Logistic regression baseline | 0.7089 |

The gap between the ensemble (best) and plain LightGBM is 0.0008 AUC — statistical noise, not a meaningful difference. Choosing plain LightGBM over the ensemble means:
- One model file to maintain instead of three
- Clean SHAP explanations (ensemble SHAP requires averaging across models)
- Simpler scoring pipeline in production
- Easier to explain to a non-technical audience

**Why hyperparameter tuning did not help**
Optuna ran 40 trials with 3-fold cross-validation and found nothing better than the defaults. This is the most important insight from the modelling phase — it means the **data is doing the work, not the model architecture**. The feature engineering across 30+ sources is what drove performance from 0.71 to 0.80. No amount of tuning would have replicated that.

---

## Results

### Model performance

| Metric | Value | Interpretation |
|---|---|---|
| ROC-AUC (random split) | 0.8008 | Correctly ranks seller above REO case 80% of the time |
| ROC-AUC (time-aware split) | 0.7365 | Performance on genuinely unseen 2023-2024 data |
| PR-AUC | 0.8125 | Strong precision-recall balance |
| Brier score | 0.1822 | Well-calibrated (random = 0.25, perfect = 0.0) |
| Accuracy at threshold 0.392 | 71.9% | Overall correct classifications |

### Top five features driving predictions (SHAP)

| Rank | Feature | Importance | Key finding |
|---|---|---|---|
| 1 | State | 0.540 | Arizona 80%+ sale rate. Some Midwest states under 30%. Geography dominates. |
| 2 | Equity ratio | 0.235 | More equity = investor sees upside = bids |
| 3 | Discount to BPO | 0.220 | Deeper auction starting price discount = more bidders |
| 4 | LTV at foreclosure | 0.213 | High debt relative to value = no investor interest |
| 5 | Property type | 0.170 | Single-family homes sell more reliably than condos or manufactured homes |

### Decile lift

| Decile | Actual sale rate |
|---|---|
| 1 (lowest score) | 9.3% |
| 2 | 22.5% |
| 3–8 | 30–71% |
| 9 | 81.8% |
| 10 (highest score) | 94.5% |

**10.2x difference** between the top and bottom decile.

### The venue finding

- Judicial vs non-judicial sale rate gap: **0.4 percentage points** (effectively zero)
- Predictions changed when flipping channel on every test case: **12 out of 4,718 (0.25%)**
- Online-heavy vs courthouse-heavy states: **3.4pp raw gap collapses to 0.18pp** after controlling for market context

Auction venue is not a meaningful lever. Market and loan fundamentals are.

### Projected business impact

| Scenario | Annual value |
|---|---|
| Conservative (5% of bottom-decile REOs prevented) | $62.8M |
| Mid (10%) | $95.6M |
| High (15%) | $128.4M |

*Based on $65K REO holding cost, 200K US GSE foreclosures/year, 30% servicer market share. Replace with actual numbers.*

---

## How to Interpret the Results

### The probability score

The model outputs a number between 0 and 1 for each case. This is the **probability of a third-party sale at auction**.

- A score of **0.91** means the model estimates a 91% chance this property will sell at auction
- A score of **0.08** means the model estimates only an 8% chance — this case is almost certainly heading toward REO

These probabilities are well-calibrated, meaning a score of 70% actually corresponds to approximately 70% of such cases selling in practice. You can use the number directly, not just as a ranking.

### The decile

Because the raw probability is harder to action on, each case gets assigned a decile (1-10) based on where its score sits relative to all other cases scored in the same batch.

- **Decile 10** = top 10% of predicted sale probability = very likely to sell
- **Decile 1** = bottom 10% = very likely to revert to REO

### The routing action

Three buckets derived from decile:

| Deciles | Sale rate in data | Recommended action |
|---|---|---|
| 9–10 | 88–95% | Auto-process. Standard auction prep. Minimal marketing investment. |
| 3–8 | 30–71% | Standard protocol. Most uncertainty. Marketing effort can move the needle here. |
| 1–2 | 9–23% | Active intervention. Early loss mitigation, mod-to-market evaluation. Begin REO operations prep now rather than waiting for the auction result. |

### A concrete example

Imagine two cases arrive in March 2026:

**Case A:** Florida, single-family, LTV at foreclosure 0.43, BPO discount 55%, hot metro (26-day DOM). Model score: **0.91, Decile 10.** Action: run standard process, this will sell.

**Case B:** Minnesota, single-family, LTV at foreclosure 0.79, no significant equity, unfavourable market conditions. Model score: **0.08, Decile 1.** Action: flag immediately for loss-mitigation review, start REO prep, do not spend marketing budget here.

The value is not that the model is always right. The value is that across thousands of cases per month, correctly separating the likely-sells from the likely-REOs at intake — rather than finding out six weeks later — changes what operations can do about it.

### The time-aware AUC caveat

The 0.80 AUC comes from a random split — cases from across 2016-2024 split randomly into train and test. The more honest production number is 0.74, from training on 2016-2022 and testing on 2023-2024 data the model had never seen. The 6-point drop reflects macro regime change (rate hikes, post-COVID dynamics). This is why quarterly retraining is recommended — as new disposition outcomes accumulate, the model stays current.

---

## Repository Structure

```
gse-foreclosure-disposition-analysis/
│
├── README.md
│
├── Foreclosure_Property_FINAL.ipynb       # Full pipeline (69 cells, Colab-ready)
│
└── deliverables/
    ├── GSE_Foreclosure_Whitepaper.docx    # 32-page technical whitepaper
    ├── GSE_Foreclosure_Whitepaper.pdf     # PDF version
    ├── GSE_Foreclosure_Deck.pptx          # 17-slide executive deck
    ├── GSE_Foreclosure_Deck.pdf           # PDF version
    └── GSE_Foreclosure_Data_Dictionary.xlsx  # Feature reference (3 sheets)
```

---

## How to Run

### Prerequisites

```bash
pip install lightgbm xgboost catboost optuna shap fredapi
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

You will also need:
- **Google Drive** with ~300 GB free (for raw GSE data)
- **FRED API key** — free at https://fred.stlouisfed.org/docs/api/api_key.html
- **Census API key** — free at https://api.census.gov/data/key_signup.html
- **Google Colab Pro** recommended — standard sessions time out during ingestion

### Execution order

Open `Foreclosure_Property_FINAL.ipynb` in Google Colab and run cells top to bottom. The notebook is resumable — each data source is cached to parquet after its first run, so restarting the runtime only adds a few minutes, not hours.

| Block | What it does | First run | Subsequent runs |
|---|---|---|---|
| Block 0 | Setup, Drive mount, API keys | 1 min | 1 min |
| Block 1A | Fannie Mae ingestion (40 quarters) | 6–12 hours | ~30 sec |
| Fix 1 | Fannie OLTV correction | ~30 min | ~30 sec |
| Block 1B | Freddie Mac ingestion (9 years) | 4–8 hours | ~30 sec |
| Block 1C | Combine sources | 2 min | 2 min |
| Block 1D + 1.5 | 30+ external sources | 45–90 min | ~30 sec |
| Harmonization | Unify columns | 2 min | 2 min |
| Block 2 + 2.5 | Feature engineering | 15 min | 15 min |
| Block 3 | EDA | 5 min | 5 min |
| Block 4 | Modelling | 10 min | 10 min |
| Block 5 | SHAP, counterfactual, decile lift | 5 min | 5 min |
| Analyses A–D | Supplementary analyses | 10 min | 10 min |

**First run: approximately 12–20 hours. Subsequent runs: approximately 60–90 minutes.**

### Scoring a new case

```python
import joblib
import pandas as pd

model    = joblib.load('models/lgbm_pooled.pkl')
encoders = joblib.load('models/encoders.pkl')

new_case = {
    'state': 'FL',
    'property_type': 'SF',
    'orig_ltv': 80,
    'credit_score': 685,
    'dti': 38,
    'ltv_at_foreclosure': 0.52,
    'equity_ratio': 0.48,
    'discount_to_bpo': 0.41,
    'mortgage_rate_30y': 6.75,         # Pull fresh from FRED
    'zhvi': 420000,                     # Pull from Zillow for this ZIP3
    # ... remaining features
}

df = pd.DataFrame([new_case])
for col, encoder in encoders.items():
    if col in df.columns:
        df[col] = encoder.transform(df[col].astype(str))

p_sale = model.predict(df)[0]
print(f'Probability of sale: {p_sale:.1%}')   # e.g. 83.7%
```

---

## Deliverables

| File | Description |
|---|---|
| `GSE_Foreclosure_Whitepaper.docx` | 32-page technical writeup covering data, methodology, results, business impact, productionization, and next steps |
| `GSE_Foreclosure_Deck.pptx` | 17-slide executive briefing in 16:9 format with embedded charts |
| `GSE_Foreclosure_Data_Dictionary.xlsx` | Three-sheet reference: all 57 features with descriptions and sources, source summary, top SHAP drivers |
| `Foreclosure_Property_FINAL.ipynb` | Cleaned end-to-end pipeline with terse markdown commentary, all outputs preserved |

---

## Caveats

**BPO is approximated**
The discount-to-BPO feature — the third strongest driver — uses FHFA state HPI applied to original property value as a proxy for broker price opinions. Real BPOs including property condition would improve this materially.

**ZIP3 aggregation**
GSE data publishes only 3-digit ZIP prefixes. All neighbourhood features are averaged across the ZIP3 area, which can span very different neighbourhoods in dense metros.

**Channel is inferred, not observed**
Judicial vs non-judicial is derived from state law, not from actual auction records. A rigorous online vs offline test requires the servicer's internal venue data.

**Macro regime sensitivity**
AUC drops from 0.80 to 0.74 on out-of-time 2023-2024 data. Quarterly retraining is necessary as macro conditions evolve.

**GSE loans only**
The model was trained on Fannie Mae and Freddie Mac conforming loans. It should not be applied to FHA, VA, or private-label portfolios without retraining.

---

## Next Steps

- Integrate real BPO records from the servicer — highest-leverage data improvement, estimated +0.02–0.04 AUC
- Test online vs offline rigorously using internal venue records
- Productionise scoring pipeline (6–11 weeks engineering effort)
- Set up quarterly retraining cadence (3–5 days per cycle)
- Extend to non-GSE portfolios

---

*For questions on methodology, contact the project author.*
