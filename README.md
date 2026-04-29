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
- Best ML model AUC = **0.797** (Logistic Regression, trained on 2.5M records)

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
| `BRFSSModeler` | Logistic regression + random forest, AUC/accuracy evaluation, feature importances |
| `BRFSSPipeline` | End-to-end orchestrator with Spark cache management and Parquet export |

---

## Technical Highlights

**Cross-year variable harmonization.** BRFSS uses rotating-core survey design — some variables alternate between even and odd years. The pipeline resolves this with a `COLUMN_ALIASES` mapping and graceful NaN-fill, then applies `F.isnan()` alongside `F.isNull()` to correctly handle the pandas → Spark type bridge (where pandas `np.nan` becomes Spark float `NaN`, not `null`).

**Production engineering standards.** All classes use Google-style docstrings, Python `logging` module (no `print()`), and type hints. The pipeline is structured to run identically in local mode or on a distributed cluster (Databricks, AWS EMR).

**ML at scale.** Models trained on 2,517,825 records with `VectorAssembler` → `StandardScaler` → classifier pipelines. The `dropna(subset=required)` approach preserves rotating-core records with partial missingness, increasing training coverage from ~8% to ~90% of the dataset.

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
| Logistic Regression | **0.7967** | 84.4% | 2,517,825 |
| Random Forest | 0.7860 | 84.4% | 2,517,825 |

**Top predictors (Random Forest):** Self-rated health (0.348) → Age (0.188) → BMI (0.151) → CHD comorbidity (0.140) → Obesity flag (0.114) → Physical activity (0.035)

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
pip install pyspark pyreadstat pandas numpy
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

> **Note:** BRFSS XPT files average ~100MB each (~800MB total for all 8 years). Files are not included in this repository due to size — download directly from CDC.

### Run

```bash
python brfss_analysis.py
```

Expected runtime: ~7–8 minutes on a standard laptop (local Spark mode).

Output saved to `./brfss_output/brfss_scored/` in Parquet format.

---

## Data Notes

**BRFSS rotating core design.** Some survey questions alternate between even and odd years:

| Variable | Available years |
|---|---|
| Hypertension (`BPHIGH4`/`BPHIGH6`) | 2017, 2019, 2021, 2023 |
| Sleep duration (`SLEPTIM1`) | 2017, 2018, 2020, 2022 |
| Kidney disease (`CHCKDNY2`) | 2018–2024 |
| Diabetes (`DIABETE4`) | All years |

Year-over-year HTN comparisons are restricted to odd years to avoid misleading 0% values caused by rotating-core structural missingness.

**Missing value codes.** BRFSS encodes non-response as trailing 7s and 9s (7, 9, 77, 99, 777, 999). These are replaced with null during cleaning. Spark float `NaN` values (from the pyreadstat → pandas → Spark bridge) are handled separately via `F.isnan()`.

**Survey weights.** Analyses use unweighted proportions. For population-representative estimates, apply `_LLCPWT` raking weights — a planned extension.

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
