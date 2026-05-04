# BRFSS 2017–2024: Lifestyle, Physical Activity & Chronic Disease Risk

A distributed data pipeline analyzing **3.48 million** CDC survey records across 8 years using Apache PySpark — examining trends in diabetes, obesity, physical activity, and chronic disease risk in the U.S. adult population.

**[View Research Report →](https://kevintan701.github.io/brfss-analysis/)**

---

## Overview

The Behavioral Risk Factor Surveillance System (BRFSS) is the CDC's annual telephone health survey, covering all 50 U.S. states with ~400,000–460,000 adult respondents per year. This project builds a production-grade PySpark pipeline to process and analyze the full 2017–2024 dataset — a scale that exceeds practical limits for single-machine tools like pandas.

**Key findings:**
- Diabetes prevalence rose from **14.2% (2017) → 15.2% (2024)**
- Obesity rose from **31.1% → 33.7%** over the same period
- Obese adults have **3.2× higher** diabetes prevalence than normal-weight adults (23.5% vs. 7.3%)
- Adults with all 4 healthy lifestyle behaviors show **6.6% diabetes rate** vs. **18.6%** for those with none
- Logistic regression AUC = **0.811** (statsmodels, N=1,060,916); RF AUC = **0.811** (PySpark, N=3.1M)

---

## Pipeline Architecture

```
BRFSSLoader → BRFSSCleaner → BRFSSEngineer → BRFSSAnalyzer → BRFSSModeler
```

| Class | Responsibility |
|---|---|
| `BRFSSConfig` | Centralized configuration — paths, column lists, hyperparameters |
| `BRFSSLoader` | XPT ingestion via pyreadstat, multi-year union, cross-year variable alias resolution |
| `BRFSSCleaner` | BRFSS missing-code handling (7/9/77/99/777/999 → null), NaN/null harmonization, type casting |
| `BRFSSEngineer` | Binary outcome flags, composite lifestyle score (0–4), disease burden index, log transforms |
| `BRFSSAnalyzer` | 10-section EDA: prevalence tables, year trends, group comparisons |
| `BRFSSModeler` | statsmodels Logit for full Wald inference (OR, SE, CI, p-value); Random Forest in PySpark for feature importance; exports Table 1 & Table 2 as CSV |
| `BRFSSPipeline` | End-to-end orchestrator: coordinates all stages, manages intermediate datasets, and exports results |

---

## Technical Highlights

**Cross-year variable harmonization.** BRFSS uses a rotating-core survey design in which some questions alternate between even and odd years. The pipeline systematically resolves cross-year naming inconsistencies and correctly distinguishes between questions that were not administered in a given year versus questions that received a non-response — ensuring that structural missingness is not conflated with item non-response.

**Production engineering standards.** All classes include full documentation, structured logging, and type hints — following conventions used in production data engineering environments. The pipeline architecture is designed to scale from a single workstation to a distributed cloud cluster without code changes.

**Inference-focused modelling.** Two feature sets are maintained: `INFERENCE_FEATURE_COLS` (13 variables available in all 8 survey years, used for statsmodels logistic regression) and `FEATURE_COLS` (16 variables including rotating-core HTN and sleep, used for Random Forest and EDA). This separation prevents listwise deletion from reducing the inference sample to ~9,000 records. PySpark handles all large-scale data operations (3.5M records). For the inference stage, the modeling subset is collected to pandas and fitted with statsmodels logistic regression — providing the complete Wald inference table (coefficient, SE, OR, 95% CI, p-value) for each predictor, equivalent to SAS PROC LOGISTIC output. Random Forest runs in PySpark as a supplementary non-parametric check. Two CSV attachments are exported: `table1_patient_characteristics.csv` and `table2_model_estimates.csv`.

---

## Results Summary

### Chronic disease prevalence (2017–2024 pooled)

| Condition | Prevalence | Trend |
|---|---|---|
| Diabetes | 14.5% | ↑ +1.0pp since 2017 |
| Obesity | 32.6% | ↑ +2.6pp since 2017 |
| CKD | 3.7% | ↑ rising (2018–2024) |
| CHD | 5.8% | Stable |
| Hypertension | ~40% | Stable (odd years only) |

### Diabetes by BMI category

| BMI Category | Diabetes % |
|---|---|
| Underweight | 6.2% |
| Normal (18.5–24.9) | 7.3% |
| Overweight (25–29.9) | 12.9% |
| Obese (≥30) | **23.5%** |

### Machine learning models

| Model | AUC-ROC | Accuracy | Train N |
|---|---|---|---|
| Logistic Regression (statsmodels) | **0.811** (RF proxy) | — | 1,060,916 (complete-case) |
| Random Forest (PySpark) | **0.811** | 84.5% | 3,147,537 |

**Top predictors (Random Forest):** Hypertension (0.233) → Self-rated health (0.221) → Age (0.166) → BMI (0.163) → CHD (0.062) → Obesity (0.042) → PA volume (0.034)

**Key inference findings (logistic regression, N=1,060,916, Pseudo-R²=0.176):** Self-rated health OR=0.611 (p<0.001) → CHD OR=1.679 (p<0.001) → Obesity OR=1.466 (p<0.001) → Age OR=1.038/yr (p<0.001) → Any PA OR=0.864 (p<0.001) → BMI OR=1.050/unit (p<0.001)

---

## Getting Started

### Requirements

- Python 3.9+
- Java 17+ (required for PySpark)

### Installation

```bash
git clone https://github.com/yourusername/brfss-analysis.git
cd brfss-analysis
python3 -m venv venv
source venv/bin/activate
pip install pyspark pyreadstat pandas numpy statsmodels
```

### Data

Download BRFSS SAS Transport Format (`.XPT`) files for 2017–2024 from the CDC:

```
https://www.cdc.gov/brfss/annual_data/annual_data.htm
```

Place files in a `data/` directory:

```
brfss-analysis/
├── data/
│   ├── LLCP2017.XPT
│   ├── LLCP2018.XPT
│   ├── ...
│   └── LLCP2024.XPT
├── brfss_analysis.py
├── index.html
└── README.md
```

> **Note:** BRFSS XPT files average ~1000MB each (~8GB total for all 8 years). Files are not included in this repository due to size — download directly from CDC.

### Run

```bash
python brfss_analysis.py
```

Expected runtime: ~7–8 minutes on a standard laptop (local Spark mode).

Output saved to:
- `./brfss_output/brfss_scored/` — scored dataset
- `./brfss_output/table1_patient_characteristics.csv` — baseline characteristics by diabetes status
- `./brfss_output/table2_model_estimates.csv` — logistic regression inference table (OR, SE, CI, p-value)

---

## Data Notes

**BRFSS rotating core design.** Some survey questions alternate between even and odd years:

| Variable | Available years |
|---|---|
| Hypertension | 2017, 2019, 2021, 2023 |
| Sleep duration | 2017, 2018, 2020, 2022 |
| Kidney disease | 2018–2024 |
| Diabetes | All years |

Year-over-year HTN comparisons are restricted to odd years to avoid misleading 0% values caused by rotating-core structural missingness.

**Missing value codes.** BRFSS encodes non-response and refusal as trailing 7s and 9s (7, 9, 77, 99, 777, 999). These are replaced with null values during the cleaning stage to ensure consistent missing data handling throughout the pipeline.

**Survey weights.** Analyses use unweighted proportions. For population-representative estimates, BRFSS raking weights should be applied — a planned extension of this work.

---

## References

1. CDC. *National Diabetes Statistics Report 2024.* US Department of Health and Human Services; 2024.
2. CDC. *BRFSS Annual Survey Data 2017–2024.* https://www.cdc.gov/brfss/annual_data/
3. Battaglia MP, et al. Improving standard poststratification techniques for RDD telephone surveys. *Surv Res Methods.* 2008;2(1):11–19.
4. Zaharia M, et al. Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. *NSDI '12.* 2012.

---

## Author

**Yuntao (Kevin) Tan**  
tyuntao@umich.edu  
March 2026
