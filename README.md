# GSE Foreclosure Disposition Analysis

**Predicting whether a foreclosed property will sell at auction (3rd-party sale) or revert to the bank as REO (Real Estate Owned)**

Prepared for ServiceLink | April 2026

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Dataset](#dataset)
- [External Data Sources](#external-data-sources)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Deliverables](#deliverables)
- [Data Dictionary](#data-dictionary)
- [Caveats](#caveats)
- [Next Steps](#next-steps)

---

## Overview

When a GSE-backed mortgage defaults and reaches foreclosure, the property goes to public auction. Two outcomes are possible:

- **Third-party sale (ZBC 02):** An external investor bids and buys the property. The bank recovers value quickly with minimal carrying cost.
- **REO reversion (ZBC 09):** No qualifying bid is received. The property reverts to the lender, who must now maintain, manage, and eventually resell it. Industry all-in cost: **$50,000–$80,000 per property.**

This project builds a machine learning model that predicts which outcome will occur — before the auction takes place — enabling ServiceLink to triage cases at intake, prioritise intervention on high-REO-risk cases, and stop wasting effort on cases that will sell regardless.

---

## The Problem

### Business question

> *Given a foreclosure case at intake, can we predict whether it will sell at auction or revert to REO — and use that prediction to make better operational decisions?*

### Why this matters

ServiceLink handles foreclosures for banks at scale. A reliable triage score at intake allows them to:

- Route **top-decile cases** (94.5% sell rate) through standard process with minimal marketing effort
- Flag **bottom-decile cases** (9.3% sell rate) for early loss-mitigation and pre-emptive REO operations
- Stop optimising **auction venue routing** — which the data shows does not meaningfully change outcomes

### What we are predicting

Binary classification: **y = 1** (third-party sale), **y = 0** (REO reversion)

---

## Dataset

### Primary source: GSE loan-performance data

| Source | Files | Raw size | Liquidations |
|---|---|---|---|
| Fannie Mae | 40 quarterly ZIPs | ~240 GB | 12,042 |
| Freddie Mac | 9 yearly ZIPs | ~40 GB | 6,829 |
| **Total** | **49 files** | **~280 GB** | **18,871** |

- Disposition window: January 2016 – December 2024
- Target balance: 51.6% sold / 48.4% REO (well-balanced; no resampling required)

Fannie Mae and Freddie Mac were chosen because they are the only public sources that provide loan-level data with verified disposition outcomes at scale, covering roughly 50% of all US conforming mortgages.

### Three critical data fixes

The raw data required three targeted patches before it was usable:

**Fix 1 — Fannie OLTV column misdetection**
The content-based column detector misidentified Fannie's interest rate column (values 2–8%) as the LTV column (values 15–97%). This left `orig_ltv` at 36% coverage for Fannie loans. A stricter whole-number detector (15–97 only, no decimals) fixed it. Coverage: 36% → 100%.

**Fix 2 — FRED distress series ID**
The original FRED series ID (`DRSFRMACBSQ`) returned empty data. Correct series: `DRSFRMACBS` (SF residential mortgage delinquency rate, all banks).

**Fix 3 — ZIP3 source-aware extraction**
Fannie pads ZIPs with leading zeros (`00360` → real ZIP3 = `360`, last 3 chars). Freddie pads with trailing zeros (`61000` → real ZIP3 = `610`, first 3 chars). A naive `zip[:3]` extraction produced Puerto Rico prefixes for all Fannie rows, causing every neighbourhood join to silently fail. Source-aware extraction fixed this. ZHVI and ACS coverage: 36% → 99%.

---

## External Data Sources

Loan-only models reach AUC ~0.71. The remaining performance comes from external context — specifically, what an investor sees when deciding whether to bid. Sources are organised into eight categories:

| Category | Sources | What it captures |
|---|---|---|
| A — Investor economics | Zillow ZHVI, ZORI, ZHVF | Rental yield, home values, price expectations |
| B — Market supply & demand | Realtor.com metro inventory, FRED supply series | Inventory, days on market, new listings |
| C — Buyer psychology | UMich sentiment (UMCSENT), inflation expectations (MICH), payrolls (PAYEMS) | Buyer confidence and employment |
| D — Local economic conditions | FHFA MSA HPI, BLS QCEW wages, IRS migration | Metro price levels, income, population flow |
| E — Credit & capital markets | ICE BofA high-yield spread, lending standards, savings rate, debt service ratios | Investor capital availability |
| F — Distress & vacancy | Rental vacancy (RRVRUSQ156N), homeowner vacancy (RHVRUSQ156N), SF delinquency (DRSFRMACBS) | Shadow inventory, sector stress |
| G — Location risk | FEMA NRI (excluded — IP-blocked from cloud egress) | Natural hazard, social vulnerability |
| H — Neighbourhood indicators | Census ACS demographics at ZIP3 level | Investor density, rent burden, education, market tightness |

**Total: 57 engineered features** across 52 numeric and 5 categorical columns.

---

## Key Findings

### 1. Geography dominates everything

State is the single most important driver (mean |SHAP| = 0.540, more than double the next feature). Arizona, Utah, and Nevada sell at 75–81%. Some Midwest states sell at under 30%. Loan characteristics matter, but geography matters more.

### 2. The five drivers that explain ~60% of outcomes

| Rank | Feature | Mean |SHAP| | What it measures |
|---|---|---|---|
| 1 | `state` | 0.540 | Geographic market dynamics |
| 2 | `equity_ratio` | 0.235 | Remaining equity — investor upside |
| 3 | `discount_to_bpo` | 0.220 | Auction starting price vs market value |
| 4 | `ltv_at_foreclosure` | 0.213 | Debt overhang at disposition |
| 5 | `property_type` | 0.170 | SF / Condo / Manufactured / Co-op |

### 3. Auction venue is not a meaningful lever

| Test | Finding |
|---|---|
| Judicial vs non-judicial sale rates | 51.4% vs 51.8% — a 0.4pp gap |
| Counterfactual channel flip | 12 of 4,718 predictions changed (0.25%) |
| Online vs offline (indirect tier test) | 3.4pp raw gap collapses to 0.18pp after controlling for market context |

The apparent advantage of "online-heavy" states is a market-selection effect, not a venue mechanic.

### 4. The decile spread is the operational lever

| Decile | Actual sale rate | Lift vs baseline |
|---|---|---|
| 1 (lowest) | 9.3% | 0.18× |
| 2 | 22.5% | 0.43× |
| 5–6 | ~53% | ~1.0× |
| 9 | 81.8% | 1.58× |
| 10 (highest) | 94.5% | 1.83× |

**10.2× spread between top and bottom decile.** Routing recommendation:
- **Deciles 9–10:** Auto-process. These will sell.
- **Deciles 3–8:** Standard protocol.
- **Deciles 1–2:** Active intervention. Begin REO operations prep now.

### 5. Six macro indicators lead the sale rate by 6–11 months

Leading indicators (negative lag = moves before sale rate):

| Indicator | Best lag | Correlation |
|---|---|---|
| Active Listings US (ACTLISCOUUS) | -10 months | r = -0.80 |
| Homeowner Vacancy (RHVRUSQ156N) | -9 months | r = -0.80 |
| Consumer Sentiment (UMCSENT) | -6 months | r = -0.74 |
| Debt Service Ratio (TDSP) | -11 months | r = -0.64 |
| Rental Vacancy (RRVRUSQ156N) | -11 months | r = -0.60 |
| Inflation Expectations (MICH) | -7 months | r = +0.59 |

These can be used as forward signals to scale triage staffing 6–11 months ahead of REO-rate spikes.

---

## Model Performance

### Results summary

| Model | ROC-AUC | Split type | Test n |
|---|---|---|---|
| LightGBM (defaults) — **selected** | **0.8008** | Random 75/25 | 4,718 |
| Ensemble (LGBM + XGB + CatBoost) | 0.8016 | Random 75/25 | 4,718 |
| LightGBM (tuned, Optuna 40 trials) | 0.8003 | Random 75/25 | 4,718 |
| CatBoost (defaults) | 0.7972 | Random 75/25 | 4,718 |
| XGBoost (defaults) | 0.7959 | Random 75/25 | 4,718 |
| Per-channel LightGBM | 0.7947 | Random 75/25 | 4,718 |
| Logistic regression baseline | 0.7089 | Random 75/25 | 4,718 |
| **Time-aware LightGBM** | **0.7365** | **Train 2016–22, test 2023–24** | **9,526** |

### Why plain LightGBM defaults

- AUC is statistically indistinguishable from tuned and ensemble variants
- Single model = simpler deployment and maintenance
- Clean SHAP support — ensemble SHAP requires per-model attribution averaging
- Optuna found nothing better than defaults in 40 trials — feature engineering drove the result

### Calibration

- **Brier score: 0.1822** (random = 0.25, perfect = 0.0)
- Model probabilities are well-calibrated — a score of 70% corresponds to ~70% observed sale rate
- Probabilities can be used directly as decision inputs, not just rankings

### Confusion matrix at threshold 0.392 (F1-optimal)

| | Predicted REO | Predicted Sold |
|---|---|---|
| **Actual REO** | 1,340 (TN) | 941 (FP) |
| **Actual Sold** | 383 (FN) | 2,054 (TP) |

Overall accuracy: 71.9%

### Dollar impact (indicative, based on industry benchmarks)

| Intervention effectiveness | Annual savings |
|---|---|
| 5% of bottom-decile REOs prevented | $62.8M |
| 10% | $95.6M |
| 15% | $128.4M |

*Assumes: $65K REO holding cost, 200K US GSE foreclosures/year, 30% ServiceLink share. Replace with internal numbers before committing.*

---

## Repository Structure

```
gse-foreclosure-disposition-analysis/
│
├── README.md                              # This file
│
├── Foreclosure_Property_FINAL.ipynb       # Full analysis pipeline (Colab-ready)
│
├── deliverables/
│   ├── GSE_Foreclosure_Whitepaper.docx    # 32-page technical whitepaper
│   ├── GSE_Foreclosure_Whitepaper.pdf     # PDF version of whitepaper
│   ├── GSE_Foreclosure_Deck.pptx          # 17-slide executive deck
│   ├── GSE_Foreclosure_Deck.pdf           # PDF version of deck
│   └── GSE_Foreclosure_Data_Dictionary.xlsx  # Feature reference (3 sheets)
│
└── docs/
    └── pipeline_overview.md               # High-level pipeline description
```

---

## How to Run

### Prerequisites

```bash
pip install lightgbm xgboost catboost optuna shap fredapi openpyxl
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

You will also need:
- **Google Drive** with ~300 GB of free space (for raw GSE data)
- **FRED API key** — free at https://fred.stlouisfed.org/docs/api/api_key.html
- **Census API key** — free at https://api.census.gov/data/key_signup.html
- **Google Colab Pro** recommended (standard sessions time out during ingestion)

### Running the notebook

Open `Foreclosure_Property_FINAL.ipynb` in Google Colab. Run cells in order:

| Block | What it does | First-run time | Subsequent runs |
|---|---|---|---|
| Block 0 | Setup, mount Drive, set API keys | 1 min | 1 min |
| Block 1A | Fannie Mae ingestion (40 quarters) | 6–12 hours | ~30 sec (cached) |
| Fix 1 | Fannie OLTV correction | ~30 min | ~30 sec (cached) |
| Block 1B | Freddie Mac ingestion (9 years) | 4–8 hours | ~30 sec (cached) |
| Block 1C | Combine sources | 2 min | 2 min |
| Block 1D | 10 original external sources | 20 min | ~5 sec (cached) |
| Block 1.5 A–H | 30 extended external sources | 30–60 min | ~5 sec (cached) |
| Harmonization | Unify columns | 2 min | 2 min |
| Block 2 | Core feature engineering | 5 min | 5 min |
| Block 2.5 | Extended feature engineering | 10 min | 10 min |
| ZIP3 fix | Source-aware ZIP extraction | 3 min | 3 min |
| LTV repair | UPB cascade fix | 1 min | 1 min |
| Block 3 | EDA and charts | 5 min | 5 min |
| Block 4 | Modelling | 10 min | 10 min |
| Block 5 | SHAP, counterfactual, decile lift | 5 min | 5 min |
| Analyses A–D | Dollar impact, subgroup, calibration, lead-lag, online-vs-offline | 10 min | 10 min |

**Total first run: approximately 12–20 hours** (mostly GSE ingestion)
**Subsequent runs: approximately 60–90 minutes** (all source files cached)

### Scoring a new 2026 case

```python
import joblib
import pandas as pd

# Load saved artifacts
model    = joblib.load('models/lgbm_pooled.pkl')
encoders = joblib.load('models/encoders.pkl')

# Assemble features for a new case
new_case = {
    'state': 'FL',
    'property_type': 'SF',
    'occupancy': 'Principal',
    'orig_ltv': 80,
    'credit_score': 685,
    'dti': 38,
    'orig_rate': 6.8,
    'ltv_at_foreclosure': 0.52,
    'equity_ratio': 0.48,
    'discount_to_bpo': 0.41,
    'months_in_default': 18,
    'mortgage_rate_30y': 6.75,        # Current from FRED
    'unemployment_rate_national': 4.2, # Current from FRED
    'zhvi': 420000,                    # Current Zillow for this ZIP3
    # ... remaining 43 features
}

df = pd.DataFrame([new_case])

# Apply encoders (categorical → numeric)
for col, encoder in encoders.items():
    if col in df.columns:
        df[col] = encoder.transform(df[col].astype(str))

# Score
p_sale = model.predict(df)[0]
print(f'Probability of 3rd-party sale: {p_sale:.2%}')
# e.g. → Probability of 3rd-party sale: 83.7%
```

---

## Deliverables

| File | Description | Pages / Slides |
|---|---|---|
| `GSE_Foreclosure_Whitepaper.docx` | Full technical writeup covering data, methodology, results, business impact, production deployment, and recommendations | 32 pages |
| `GSE_Foreclosure_Deck.pptx` | Executive briefing in 16:9 format | 17 slides |
| `GSE_Foreclosure_Data_Dictionary.xlsx` | Three-sheet reference: full feature table, source summary, top drivers with SHAP values | 3 sheets |
| `Foreclosure_Property_FINAL.ipynb` | Cleaned, commented end-to-end pipeline. All outputs preserved from original run. | 69 cells |

---

## Data Dictionary

Full details in `GSE_Foreclosure_Data_Dictionary.xlsx`. Summary of feature categories:

| Category | Features | Coverage |
|---|---|---|
| Loan fundamentals | orig_ltv, dti, credit_score, ltv_at_foreclosure, discount_to_bpo, months_in_default, equity_ratio, orig_rate | 70–100% |
| Geographic & housing market | hpi_orig, hpi_disp, hpi_msa_disp, market_friction_score | 90–100% |
| Macro at disposition | mortgage_rate_30y, unemployment_rate_national, mortgage_rate_delta_90d | 99–100% |
| Buyer psychology | UMCSENT, MICH, PAYEMS | 99–100% |
| Supply indicators | MSACSR, ACTLISCOUUS, HOUST1F, COMPU1USA | 99–100% |
| Realtor.com metro | active_listings_final, median_dom_final, new_listings_final, median_list_price_final | ~85% (national fallback) |
| Credit & capital | BAMLH0A0HYM2, DRTSCLCC, PSAVERT, TDSP, MDSP, CORESTICKM159SFRBATL, HSN1F | 95–99% |
| Distress & vacancy | RRVRUSQ156N, RHVRUSQ156N, DRSFRMACBS | 95–99% |
| ZIP-level home market | zhvi, zori, rent_to_price, zhvf_latest_forecast | 70–99% |
| ACS demographics | median_hh_income, pct_owner_occupied, poverty_rate | 99% |
| Neighbourhood proxies | investor_density_proxy, mortgage_leverage_ratio, shadow_inventory_proxy, rent_burden, pct_new_construction, education_index, market_tightness_proxy | 99% |
| State economy | state_wkly_wage, state_total_inflow | 100% |
| Categorical | state, property_type, occupancy, channel_proxy, income_tier | 100% |

---

## Caveats

**BPO is approximated, not observed**
Discount to BPO — the third strongest feature — uses FHFA state-level HPI applied to original property value as a proxy. Real broker price opinions including property condition and recent comparables would improve this materially.

**ZIP3 aggregation dilutes neighbourhood signal**
GSE data publishes only 3-digit ZIP prefixes. All neighbourhood features are aggregated to ZIP3 by mean, which averages across heterogeneous neighbourhoods.

**Channel is inferred, not observed**
Judicial vs non-judicial is inferred from state law, not from actual auction-venue records. A definitive online vs offline test requires ServiceLink's internal venue data.

**Time-aware AUC indicates regime sensitivity**
AUC drops from 0.80 to 0.74 on 2023–2024 out-of-time data. Quarterly retraining is necessary to maintain accuracy as macro conditions evolve.

**GSE loans only**
The model was trained on Fannie Mae and Freddie Mac loans. It should not be applied to FHA, VA, or private-label loans without retraining on those populations.

---

## Next Steps

### Short-term (30–90 days)
- Validate cost assumptions internally — replace $65K REO holding cost and 30% market share with actual ServiceLink numbers
- Pilot decile triage on 6–8 weeks of incoming cases without changing operations
- Identify the BPO data integration owner — highest-leverage data improvement (+0.02–0.04 AUC)

### Medium-term (3–6 months)
- Productionise the model with quarterly retraining cadence
- Stand up a triage workflow tied to the score
- Replace BPO proxy with real broker price opinions

### Long-term (6–12 months)
- Test online vs offline rigorously using ServiceLink's internal venue records
- Extend to non-GSE portfolios (FHA, VA, private-label)
- Add full property-characteristic data (ATTOM, CoreLogic, MLS)
- Build an origination-time disposition risk model

---

## Production Deployment

### Scoring new cases

The trained model (`lgbm_pooled.pkl`) can score any incoming foreclosure case in under one second. The scoring pipeline requires:

| Component | Function | Effort |
|---|---|---|
| Intake adapter | Pulls new cases from ServiceLink's system into the feature schema | 1–2 weeks |
| Feature joiner | Pulls fresh macro and market context from FRED / Zillow / Realtor.com | 1–2 weeks |
| Scoring service | REST API or batch job wrapping the model | 1 week |
| Monitoring | Score distribution tracking, feature coverage alerts | 1–2 weeks |
| UI integration | Surfaces score and decile to case managers | 2–4 weeks |

**Total engineering: 6–11 weeks**

### Quarterly retraining

Each retrain cycle (3–5 days for a data scientist):
1. Pull new GSE disposition data (Fannie + Freddie publish quarterly)
2. Refresh external sources (FRED, Zillow, Realtor.com, Census ACS)
3. Rebuild feature matrix
4. Retrain LightGBM with same hyperparameters
5. Validate on most recent quarter holdout before promoting to production

### Team composition

| Role | Ownership |
|---|---|
| Data scientist (1 FTE) | Quarterly retrains, validation, model improvements |
| ML engineer (0.5–1 FTE) | Scoring pipeline, monitoring, deployment |
| Data engineer (0.5 FTE) | External data refresh jobs |
| Product / ops owner | Triage workflow, business impact measurement |

---

*Technical whitepaper, executive deck, and data dictionary available in the `deliverables/` folder.*
