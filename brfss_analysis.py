# =============================================================================
#  PROJECT:  Behavioral Risk Factor Surveillance System (BRFSS) 2017–2024
#  PROGRAM:  brfss_analysis.py
#  AUTHOR:   Yuntao (Kevin) Tan
#  DATE:     March 2026
#
#  DESCRIPTION:
#    End-to-end PySpark pipeline for multi-year BRFSS analysis (2017–2024).
#    Combines 8 years of CDC survey data (~3.5M records) to examine trends
#    in physical activity, sleep, mental health, and chronic disease risk
#    across the U.S. adult population.
#
#  DATA SOURCE:
#    CDC Behavioral Risk Factor Surveillance System (BRFSS)
#    https://www.cdc.gov/brfss/annual_data/annual_data.htm
#    Files: LLCP2017.XPT ~ LLCP2024.XPT (SAS Transport Format)
#
#  PIPELINE STRUCTURE:
#    BRFSSConfig      — Central configuration (paths, columns, constants)
#    BRFSSLoader      — XPT ingestion and Spark DataFrame creation
#    BRFSSCleaner     — Missing value handling, type casting, standardization
#    BRFSSEngineer    — Feature engineering and derived variables
#    BRFSSAnalyzer    — EDA: descriptive stats, group comparisons, trends
#    BRFSSModeler     — Inference (statsmodels) + RF feature importance (PySpark)
#    BRFSSPipeline    — Orchestrator: runs all stages end-to-end
#
#  REQUIREMENTS:
#    pip install pyspark pyreadstat pandas numpy statsmodels
#    Java 17+ required for PySpark
#
#  USAGE:
#    python brfss_analysis.py
# =============================================================================

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

import statsmodels.api as sm


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("brfss")


# =============================================================================
# SECTION 0: CONFIGURATION
# =============================================================================

@dataclass
class BRFSSConfig:
    """
    Central configuration for the BRFSS analysis pipeline.

    All file paths, column selections, and analysis constants are
    defined here to make the pipeline easy to reconfigure without
    touching analysis code.

    Attributes
    ----------
    data_dir : str
        Directory containing LLCP20XX.XPT files.
    output_dir : str
        Directory for Parquet output and scored datasets.
    years : List[int]
        Survey years to include in the analysis.
    seed : int
        Random seed for reproducibility.
    train_ratio : float
        Fraction of data used for model training (remainder = test).

    Notes
    -----
    BRFSS missing value codes:
        Blank / 7 / 77 / 777  → "don't know"
        9 / 99 / 999           → "refused" or "not asked"
    Both are treated as null in cleaning.
    """

    data_dir: str = "./data"
    output_dir: str = "./brfss_output"
    years: List[int] = field(
        default_factory=lambda: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    )
    seed: int = 20260301
    train_ratio: float = 0.80

    # ── Core variables to retain after loading ──────────────────────────────
    # CDC pre-computed variables (leading underscore) are preferred over raw
    # source variables because they already handle skip patterns and recodes.
    core_columns: List[str] = field(default_factory=lambda: [
        # Survey metadata
        "IYEAR",        # Survey year
        "_STATE",       # State FIPS code

        # Demographics
        "SEXVAR",       # Sex (2017-2024 harmonized name; fallback: SEX1)
        "_AGE80",       # Age in years (80+ collapsed for privacy)
        "_IMPRACE",     # Imputed race/ethnicity
        "EDUCA",        # Education level
        "INCOME3",      # Household income (2019+ name; fallback: INCOME2)

        # Physical activity  (CDC pre-computed)
        "_TOTINDA",     # Any physical activity past 30 days (1=yes, 2=no)
        "PA1MIN_",      # Total PA minutes/week (computed by CDC)

        # Sleep
        "SLEPTIM1",     # Hours of sleep per night (1-24; 77/99=missing)

        # Mental & physical health
        "MENTHLTH",     # Days mental health not good past 30 days
        "PHYSHLTH",     # Days physical health not good past 30 days
        "GENHLTH",      # General health self-rating (1=Excellent … 5=Poor)

        # Body measures
        "_BMI5",        # BMI × 100 (e.g. 2500 = BMI 25.0)
        "_BMI5CAT",     # BMI category (1=Underweight … 4=Obese)

        # Chronic conditions (primary outcomes)
        "DIABETE4",     # Diabetes diagnosis (1=Yes, 2=Yes-pregnancy, 3=No, 4=Pre)
        "BPHIGH6",      # High blood pressure ever (1=Yes, 2=No; pre-2019: BPHIGH4)
        "CVDCRHD4",     # Coronary heart disease (1=Yes, 2=No)
        "CHCKDNY2",     # Kidney disease (1=Yes, 2=No)
        "_RFCHOL3",     # High cholesterol (1=No, 2=Yes; computed)

        # Lifestyle
        "SMOKE100",     # Smoked ≥100 cigarettes lifetime (1=Yes, 2=No)
        "ALCDAY4",      # Days drinking alcohol past 30 days
        "EXERHMM1",     # Minutes of vigorous activity per session

        # Survey weight
        "_LLCPWT",      # Final raking weight — use for population estimates
    ])

    # Missing value codes to replace with null (per BRFSS documentation)
    missing_codes: Dict[str, List] = field(default_factory=lambda: {
        "single_digit":  [7, 9],
        "double_digit":  [77, 99],
        "triple_digit":  [777, 999],
    })


# =============================================================================
# SECTION 1: DATA LOADER
# =============================================================================

class BRFSSLoader:
    """
    Loads BRFSS XPT files for multiple survey years and creates a
    combined Spark DataFrame.

    The loader reads each XPT file via pyreadstat (pandas bridge),
    selects the configured core columns (with graceful fallback for
    columns that changed names across years), adds a survey year tag,
    and converts to a Spark DataFrame before combining all years via
    union.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    config : BRFSSConfig
        Pipeline configuration.
    """

    # Variable name aliases: some columns were renamed across survey years.
    # Key = canonical name used internally; Value = list of possible raw names.
    COLUMN_ALIASES: Dict[str, List[str]] = {
        "SEXVAR":   ["SEXVAR", "SEX1", "SEX", "_SEX"],
        "INCOME3":  ["INCOME3", "INCOME2"],
        "BPHIGH6":  ["BPHIGH6", "BPHIGH4"],
        "CHCKDNY2": ["CHCKDNY2", "CHCKDNY1"],
        "DIABETE4": ["DIABETE4", "DIABETE3"],
        "_RFCHOL3": ["_RFCHOL3", "_RFCHOL2"],
        "SLEPTIM1": ["SLEPTIM1"],
    }

    def __init__(self, spark: SparkSession, config: BRFSSConfig) -> None:
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger("brfss.loader")

    def _resolve_column(self, df_cols: List[str], canonical: str) -> Optional[str]:
        """
        Return the actual column name in a DataFrame for a given canonical name.

        Checks the COLUMN_ALIASES mapping first; falls back to the canonical
        name itself if no alias is found or none of the aliases are present.

        Parameters
        ----------
        df_cols : List[str]
            Column names present in the raw DataFrame.
        canonical : str
            The internal canonical variable name.

        Returns
        -------
        str or None
            Actual column name to use, or None if not found anywhere.
        """
        aliases = self.COLUMN_ALIASES.get(canonical, [canonical])
        for alias in aliases:
            if alias in df_cols:
                return alias
        return None

    def _load_single_year(self, year: int) -> Optional[pd.DataFrame]:
        """
        Read one BRFSS XPT file into a pandas DataFrame.

        Selects only the configured core columns (resolving aliases),
        adds a ``survey_year`` column, and returns the result. Columns
        that are absent in a particular year's file are silently skipped
        and filled with NaN so the multi-year union remains consistent.

        Parameters
        ----------
        year : int
            Four-digit survey year (e.g. 2019).

        Returns
        -------
        pd.DataFrame or None
            Filtered DataFrame for the year, or None if the file is missing.
        """
        filepath = os.path.join(self.config.data_dir, f"LLCP{year}.XPT")

        if not Path(filepath).exists():
            self.logger.warning("File not found, skipping: %s", filepath)
            return None

        self.logger.info("Reading %s ...", filepath)
        t0 = time.time()
        df_raw, _ = pyreadstat.read_xport(filepath, encoding="latin1")
        elapsed = time.time() - t0
        self.logger.info(
            "  Loaded %s: %d rows × %d cols in %.1fs",
            f"LLCP{year}.XPT", len(df_raw), len(df_raw.columns), elapsed,
        )

        raw_cols = df_raw.columns.tolist()
        selected = {}

        for canonical in self.config.core_columns:
            actual = self._resolve_column(raw_cols, canonical)
            if actual is None:
                self.logger.debug(
                    "  Column '%s' not found in %d data — will be NaN", canonical, year
                )
                selected[canonical] = np.nan
            elif actual != canonical:
                self.logger.debug(
                    "  Alias resolved: '%s' → '%s' for year %d", canonical, actual, year
                )
                selected[canonical] = df_raw[actual]
            else:
                selected[canonical] = df_raw[canonical]

        year_survey_df = pd.DataFrame(selected)
        year_survey_df["survey_year"] = year
        return year_survey_df

    def load_all_years(self) -> DataFrame:
        """
        Load and combine BRFSS data for all configured survey years.

        Iterates over ``config.years``, loads each XPT file, and
        unions the results into a single Spark DataFrame. Years with
        missing files are skipped with a warning.

        Returns
        -------
        pyspark.sql.DataFrame
            Combined multi-year BRFSS dataset.

        Raises
        ------
        RuntimeError
            If no XPT files are successfully loaded.
        """
        self.logger.info(
            "Loading BRFSS data for years: %s", self.config.years
        )
        frames: List[pd.DataFrame] = []

        for year in self.config.years:
            year_survey_df = self._load_single_year(year)
            if year_survey_df is not None:
                frames.append(year_survey_df)

        if not frames:
            raise RuntimeError(
                "No BRFSS XPT files found. Check config.data_dir: "
                f"'{self.config.data_dir}'"
            )

        self.logger.info("Concatenating %d year(s) of data ...", len(frames))
        combined_pd = pd.concat(frames, ignore_index=True)
        self.logger.info(
            "Combined pandas shape: %d rows × %d cols",
            len(combined_pd), len(combined_pd.columns),
        )

        self.logger.info("Converting to Spark DataFrame ...")
        combined_spark_df = self.spark.createDataFrame(combined_pd)
        total = combined_spark_df.count()
        self.logger.info("Spark DataFrame created: %d total records", total)
        return combined_spark_df


# =============================================================================
# SECTION 2: DATA CLEANER
# =============================================================================

class BRFSSCleaner:
    """
    Cleans and standardizes a raw BRFSS Spark DataFrame.

    Responsibilities:
    - Replace BRFSS-specific missing value codes (7/9/77/99/777/999)
      with null.
    - Cast columns to appropriate numeric types.
    - Apply variable-specific business rules (e.g. BMI division by 100,
      reversing the GENHLTH scale so higher = better).
    - Filter to adults only and drop rows missing the primary outcome.

    Parameters
    ----------
    config : BRFSSConfig
        Pipeline configuration (provides missing code definitions).
    """

    def __init__(self, config: BRFSSConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("brfss.cleaner")

    def _nullify_missing_codes(self, survey_df: DataFrame, col: str) -> DataFrame:
        """
        Replace BRFSS refused/don't-know codes with null for one column.

        BRFSS encodes non-responses as trailing 7s and 9s (e.g. 7, 9,
        77, 99, 777, 999). This method nullifies those values based on
        the column's expected value range (inferred from column name
        conventions and value magnitude).

        Parameters
        ----------
        survey_df : DataFrame
        col : str
            Column name to process.

        Returns
        -------
        DataFrame
            DataFrame with missing codes replaced by null in ``col``.
        """
        # Determine which missing codes to apply based on observed max value.
        # We check all three tiers; Spark evaluates lazily so this is safe.
        return survey_df.withColumn(
            col,
            F.when(F.col(col).isin(
                self.config.missing_codes["single_digit"]
                + self.config.missing_codes["double_digit"]
                + self.config.missing_codes["triple_digit"]
            ), None).otherwise(F.col(col))
        )

    def standardize_survey_data(self, raw_survey_df: DataFrame) -> DataFrame:
        """
        Apply all cleaning transformations to the raw BRFSS DataFrame.

        Steps:
        1. Cast all numeric columns to DoubleType for consistent arithmetic.
        2. Replace missing value codes with null across all core columns.
        3. Apply variable-specific transformations (BMI scaling, GENHLTH
           reversal, sleep and PA hour limits).
        4. Add a ``state_name`` string label for selected states.

        Parameters
        ----------
        raw_survey_df : DataFrame
            Raw combined BRFSS DataFrame from BRFSSLoader.

        Returns
        -------
        DataFrame
            Cleaned DataFrame with standardized column types and null-coded
            missing values.
        """
        self.logger.info("Starting data cleaning ...")
        t0 = time.time()
        survey_df = raw_survey_df

        # ── Step 1: Cast to DoubleType ──────────────────────────────────────
        numeric_cols = [c for c in survey_df.columns if c != "state_name"]
        for col in numeric_cols:
            if survey_df.schema[col].dataType not in (DoubleType(), IntegerType()):
                survey_df = survey_df.withColumn(col, F.col(col).cast(DoubleType()))

        # ── Step 2: Replace missing codes ───────────────────────────────────
        missing_flag_cols = [
            "SLEPTIM1", "MENTHLTH", "PHYSHLTH", "GENHLTH", "EDUCA",
            "INCOME3", "SMOKE100", "ALCDAY4", "DIABETE4", "BPHIGH6",
            "CVDCRHD4", "CHCKDNY2", "SEXVAR", "EXERHMM1", "PA1MIN_",
        ]
        for col in missing_flag_cols:
            if col in survey_df.columns:
                survey_df = self._nullify_missing_codes(survey_df, col)

        # ── Step 3: Variable-specific transformations ────────────────────────

        # BMI: stored as integer × 100 (e.g. 2500 → 25.0 kg/m²)
        if "_BMI5" in survey_df.columns:
            survey_df = survey_df.withColumn("bmi", F.col("_BMI5") / 100.0)
            survey_df = survey_df.withColumn(
                "bmi",
                F.when((F.col("bmi") < 10) | (F.col("bmi") > 80), None)
                 .otherwise(F.col("bmi"))
            )

        # Sleep: valid range 1-24 hours
        if "SLEPTIM1" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "sleep_hrs",
                F.when(
                    (F.col("SLEPTIM1") >= 1) & (F.col("SLEPTIM1") <= 24),
                    F.col("SLEPTIM1")
                ).otherwise(None)
            )

        # GENHLTH: 1=Excellent … 5=Poor → reverse so higher = healthier
        if "GENHLTH" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "gen_health",
                F.when(F.col("GENHLTH").between(1, 5), 6 - F.col("GENHLTH"))
                 .otherwise(None)
            )  # Now 5=Excellent, 1=Poor

        # Mental and physical health days (0-30)
        for raw, clean in [("MENTHLTH", "mental_hlth_days"), ("PHYSHLTH", "phys_hlth_days")]:
            if raw in survey_df.columns:
                survey_df = survey_df.withColumn(
                    clean,
                    F.when(
                        (F.col(raw) >= 0) & (F.col(raw) <= 30),
                        F.col(raw).cast(DoubleType())
                    ).otherwise(None)
                )

        # Physical activity minutes/week (cap at 10,000 as outlier guard)
        if "PA1MIN_" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "pa_min_week",
                F.when(
                    (F.col("PA1MIN_") >= 0) & (F.col("PA1MIN_") <= 10000),
                    F.col("PA1MIN_")
                ).otherwise(None)
            )

        # Meets CDC PA guidelines: ≥150 min moderate / ≥75 min vigorous per week
        # PA1MIN_ is already in total equivalent minutes (vigorous × 2 + moderate)
        if "PA1MIN_" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "meets_pa_guidelines",
                F.when(F.col("PA1MIN_").isNull(), None)
                 .when(F.col("PA1MIN_") >= 150, F.lit(1))
                 .otherwise(F.lit(0))
            )

        # Sex: standardize to 1=Male, 2=Female
        if "SEXVAR" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "sex",
                F.when(F.col("SEXVAR").isin([1, 2]), F.col("SEXVAR").cast(IntegerType()))
                 .otherwise(None)
            )
            survey_df = survey_df.withColumn(
                "sex_label",
                F.when(F.col("sex") == 1, "Male")
                 .when(F.col("sex") == 2, "Female")
                 .otherwise(None)
            )

        # Age groups
        if "_AGE80" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "age_group",
                F.when(F.col("_AGE80") < 30,  "18–29")
                 .when(F.col("_AGE80") < 45,  "30–44")
                 .when(F.col("_AGE80") < 60,  "45–59")
                 .when(F.col("_AGE80") < 75,  "60–74")
                 .otherwise("75+")
            )

        # Physical activity category
        if "_TOTINDA" in survey_df.columns:
            survey_df = survey_df.withColumn(
                "pa_any",
                F.when(F.col("_TOTINDA") == 1, 1)
                 .when(F.col("_TOTINDA") == 2, 0)
                 .otherwise(None)
            )

        elapsed = time.time() - t0
        self.logger.info("Cleaning complete in %.1fs.", elapsed)
        return survey_df


# =============================================================================
# SECTION 3: FEATURE ENGINEER
# =============================================================================

class BRFSSEngineer:
    """
    Derives composite features and outcome flags from cleaned BRFSS data.

    Creates:
    - Binary outcome flags for diabetes, hypertension, CKD, and CHD.
    - A composite chronic disease burden score (0–4).
    - A healthy lifestyle score (0–4) combining PA, sleep, BMI, and smoking.
    - Sleep adequacy and mental health burden flags.
    - Log-transformed skewed variables for modeling.

    Parameters
    ----------
    config : BRFSSConfig
        Pipeline configuration.
    """

    def __init__(self, config: BRFSSConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("brfss.engineer")

    def _make_binary_outcome(
        self, survey_df: DataFrame, raw_col: str, new_col: str, yes_codes: List[int]
    ) -> DataFrame:
        """
        Create a binary (0/1) outcome flag from a BRFSS categorical variable.

        Parameters
        ----------
        survey_df : DataFrame
        raw_col : str
            Source column containing the BRFSS response codes.
        new_col : str
            Name of the new binary flag column.
        yes_codes : List[int]
            Response codes that map to 1 (positive outcome).
            All other non-null values map to 0.

        Returns
        -------
        DataFrame
        """
        if raw_col not in survey_df.columns:
            self.logger.debug(
                "Column '%s' not found; '%s' will be null.", raw_col, new_col
            )
            return survey_df.withColumn(new_col, F.lit(None).cast(IntegerType()))

        return survey_df.withColumn(
            new_col,
            # NaN (from pandas NaN-fill for missing rotating-core years) and
            # null are both treated as missing. isNull() catches Spark nulls;
            # isnan() catches float NaN values introduced by the pandas bridge.
            F.when(F.col(raw_col).isNull() | F.isnan(F.col(raw_col)), None)
             .when(F.col(raw_col).isin(yes_codes), F.lit(1))
             .otherwise(F.lit(0))
        )

    def build_analytic_features(self, cleaned_survey_df: DataFrame) -> DataFrame:
        """
        Apply all feature engineering transformations.

        Parameters
        ----------
        cleaned_survey_df : DataFrame
            Cleaned BRFSS DataFrame from BRFSSCleaner.

        Returns
        -------
        DataFrame
            DataFrame with engineered outcome flags, composite scores,
            and log-transformed variables appended.
        """
        self.logger.info("Starting feature engineering ...")
        t0 = time.time()
        feature_df = cleaned_survey_df

        # ── Binary outcome flags ─────────────────────────────────────────────
        feature_df = self._make_binary_outcome(feature_df, "DIABETE4",  "flag_diabetes",  [1, 2])
        feature_df = self._make_binary_outcome(feature_df, "BPHIGH6",   "flag_htn",       [1])
        feature_df = self._make_binary_outcome(feature_df, "CVDCRHD4",  "flag_chd",       [1])
        feature_df = self._make_binary_outcome(feature_df, "CHCKDNY2",  "flag_ckd",       [1])

        # Obesity flag: BMI ≥ 30
        feature_df = feature_df.withColumn(
            "flag_obese",
            F.when(F.col("bmi").isNull(), None)
             .when(F.col("bmi") >= 30, F.lit(1))
             .otherwise(F.lit(0))
        )

        # Poor sleep flag: < 7 hours (CDC recommended minimum)
        feature_df = feature_df.withColumn(
            "flag_poor_sleep",
            F.when(F.col("sleep_hrs").isNull(), None)
             .when(F.col("sleep_hrs") < 7, F.lit(1))
             .otherwise(F.lit(0))
        )

        # Mental health burden flag: ≥ 14 poor mental health days/month
        feature_df = feature_df.withColumn(
            "flag_mental_burden",
            F.when(F.col("mental_hlth_days").isNull(), None)
             .when(F.col("mental_hlth_days") >= 14, F.lit(1))
             .otherwise(F.lit(0))
        )

        # ── Composite chronic disease burden score (0–4) ─────────────────────
        # Count of the four major conditions: diabetes, hypertension, CKD, CHD
        # disease_burden: sum of confirmed conditions.
        # flag_htn is null for even survey years (BRFSS rotating core).
        # We treat null as 0 only for diabetes/CKD/CHD which are available
        # every year. flag_htn is excluded from coalesce so that even-year
        # rows are not incorrectly scored as HTN-negative.
        feature_df = feature_df.withColumn(
            "disease_burden",
            F.coalesce(F.col("flag_diabetes"), F.lit(0))
            + F.coalesce(F.col("flag_htn"),     F.lit(0))
            + F.coalesce(F.col("flag_ckd"),     F.lit(0))
            + F.coalesce(F.col("flag_chd"),     F.lit(0))
        )
        # flag_htn_available: 1 if HTN data was collected this survey year
        feature_df = feature_df.withColumn(
            "flag_htn_available",
            F.when(F.col("flag_htn").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )

        # ── Healthy lifestyle score (0–4) ────────────────────────────────────
        # +1 for meeting PA guidelines
        # +1 for adequate sleep (≥7 hrs)
        # +1 for healthy BMI (18.5–24.9)
        # +1 for never-smoker or former smoker (SMOKE100 = 2 means never)
        feature_df = feature_df.withColumn(
            "lifestyle_score",
            F.coalesce(F.col("meets_pa_guidelines"), F.lit(0))
            + F.coalesce(
                F.when(F.col("sleep_hrs") >= 7, F.lit(1))
                 .when(F.col("sleep_hrs").isNotNull(), F.lit(0)),
                F.lit(0)
              )  # null sleep_hrs (rotating core years) treated as 0, not penalised
            + F.when(
                F.col("bmi").between(18.5, 24.9), F.lit(1)
              ).otherwise(F.lit(0))
            + F.when(
                F.col("SMOKE100") == 2, F.lit(1)
              ).otherwise(F.lit(0))
        )

        # ── BMI category label ───────────────────────────────────────────────
        feature_df = feature_df.withColumn(
            "bmi_cat",
            F.when(F.col("bmi") < 18.5, "Underweight")
             .when(F.col("bmi") < 25.0, "Normal")
             .when(F.col("bmi") < 30.0, "Overweight")
             .when(F.col("bmi").isNotNull(), "Obese")
             .otherwise(None)
        )

        # ── Log transforms for skewed continuous variables ───────────────────
        feature_df = feature_df.withColumn(
            "log_pa_min",
            F.when(F.col("pa_min_week") > 0, F.log1p(F.col("pa_min_week")))
             .otherwise(F.lit(0.0))
        )

        # ── Year-over-year trend column ──────────────────────────────────────
        feature_df = feature_df.withColumn(
            "years_since_2017",
            (F.col("survey_year") - 2017).cast(DoubleType())
        )

        elapsed = time.time() - t0
        self.logger.info("Feature engineering complete in %.1fs.", elapsed)
        return feature_df


# =============================================================================
# SECTION 4: ANALYZER
# =============================================================================

class BRFSSAnalyzer:
    """
    Exploratory data analysis on the engineered BRFSS dataset.

    Computes and logs:
    - Sample overview (N, demographics, outcome prevalence).
    - Year-over-year trend in key outcomes.
    - Physical activity and sleep distributions.
    - Chronic disease prevalence by PA level, sleep, BMI category.
    - State-level outcome summaries (top/bottom 10).

    All outputs are printed to the logger at INFO level so they can
    be captured in log files for reporting.

    Parameters
    ----------
    config : BRFSSConfig
        Pipeline configuration.
    """

    def __init__(self, config: BRFSSConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("brfss.analyzer")

    def _print_summary_table(self, summary_df: DataFrame, title: str, n: int = 30) -> None:
        """
        Log a Spark DataFrame as a formatted table.

        Parameters
        ----------
        summary_df : DataFrame
        title : str
            Section header printed before the table.
        n : int
            Maximum number of rows to display.
        """
        self.logger.info("── %s ──", title)
        summary_df.show(n, truncate=False)

    def compute_descriptive_statistics(self, analytic_df: DataFrame) -> None:
        """
        Execute the full EDA suite and log results.

        Parameters
        ----------
        analytic_df : DataFrame
            Engineered BRFSS DataFrame.
        """
        self.logger.info("=" * 60)
        self.logger.info("SECTION 4: Exploratory Data Analysis")
        self.logger.info("=" * 60)

        # ── 4.1 Overall sample overview ──────────────────────────────────────
        total_n = analytic_df.count()
        self.logger.info("Total records across all years: %d", total_n)

        self._print_summary_table(
            analytic_df.select(
                F.count("*").alias("N"),
                F.round(F.mean("_AGE80"), 1).alias("mean_age"),
                F.round(F.mean("bmi"), 1).alias("mean_bmi"),
                F.round(F.mean("sleep_hrs"), 1).alias("mean_sleep_hrs"),
                F.round(F.mean("pa_min_week"), 0).alias("mean_pa_min_wk"),
                F.round(F.mean("mental_hlth_days"), 1).alias("mean_mental_bad_days"),
            ),
            "4.1 Sample Overview"
        )

        # ── 4.2 Records per survey year ──────────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("survey_year")
              .agg(F.count("*").alias("N"))
              .orderBy("survey_year"),
            "4.2 Records by Survey Year"
        )

        # ── 4.3 Chronic disease prevalence (overall) ─────────────────────────
        self._print_summary_table(
            analytic_df.select(
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")      * 100, 1).alias("Hypertension_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("flag_chd")      * 100, 1).alias("CHD_%"),
                F.round(F.mean("flag_obese")    * 100, 1).alias("Obesity_%"),
            ),
            "4.3 Chronic Disease Prevalence (Overall)"
        )

        # ── 4.4 Year-over-year trends in key outcomes ────────────────────────
        # Note: HTN data only available in odd years (BRFSS rotating core).
        # Sleep data only available in 2017/2018/2020/2022 (rotating core).
        # HTN_% shows NULL for even years — this is expected, not a data error.
        self._print_summary_table(
            analytic_df.groupBy("survey_year").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.when(
                    F.sum("flag_htn_available") > 0,
                    F.round(
                        F.sum(F.when(F.col("flag_htn") == 1, 1).otherwise(0))
                        / F.sum(F.when(F.col("flag_htn").isNotNull(), 1).otherwise(0)) * 100, 1)
                ).otherwise(F.lit(None)).alias("HTN_%_oddYrsOnly"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("flag_obese")    * 100, 1).alias("Obesity_%"),
                F.round(F.mean("pa_any")        * 100, 1).alias("Any_PA_%"),
                F.round(F.mean("sleep_hrs"),    1).alias("Mean_Sleep_availYrs"),
            ).orderBy("survey_year"),
            "4.4 Year-over-Year Trends (2017–2024) — HTN odd yrs only; Sleep 2017/18/20/22 only"
        )

        # ── 4.5 Outcomes by physical activity level ──────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("meets_pa_guidelines").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")      * 100, 1).alias("HTN_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("flag_obese")    * 100, 1).alias("Obesity_%"),
                F.round(F.mean("bmi"),          1).alias("Mean_BMI"),
                F.round(F.mean("sleep_hrs"),    1).alias("Mean_Sleep"),
            ).orderBy("meets_pa_guidelines"),
            "4.5 Outcomes by PA Guideline Adherence (0=No, 1=Yes)"
        )

        # ── 4.6 Outcomes by sleep adequacy ───────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("flag_poor_sleep").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes")      * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")           * 100, 1).alias("HTN_%"),
                F.round(F.mean("flag_ckd")           * 100, 1).alias("CKD_%"),
                F.round(F.mean("flag_mental_burden") * 100, 1).alias("MentalBurden_%"),
            ).orderBy("flag_poor_sleep"),
            "4.6 Outcomes by Sleep Adequacy (0=≥7hrs, 1=<7hrs)"
        )

        # ── 4.7 Outcomes by BMI category ─────────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("bmi_cat").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")      * 100, 1).alias("HTN_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("pa_any")        * 100, 1).alias("Any_PA_%"),
            ).orderBy("bmi_cat"),
            "4.7 Outcomes by BMI Category"
        )

        # ── 4.8 Lifestyle score distribution ─────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("lifestyle_score").agg(
                F.count("*").alias("N"),
                F.round(F.count("*") / total_n * 100, 1).alias("Pct"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
            ).orderBy("lifestyle_score"),
            "4.8 Disease Prevalence by Healthy Lifestyle Score (0–4)"
        )

        # ── 4.9 Sex breakdown ─────────────────────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("sex_label").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")      * 100, 1).alias("HTN_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("bmi"),          1).alias("Mean_BMI"),
            ).orderBy("sex_label"),
            "4.9 Outcomes by Sex"
        )

        # ── 4.10 Age group breakdown ─────────────────────────────────────────
        self._print_summary_table(
            analytic_df.groupBy("age_group").agg(
                F.count("*").alias("N"),
                F.round(F.mean("flag_diabetes") * 100, 1).alias("Diabetes_%"),
                F.round(F.mean("flag_htn")      * 100, 1).alias("HTN_%"),
                F.round(F.mean("flag_ckd")      * 100, 1).alias("CKD_%"),
                F.round(F.mean("flag_obese")    * 100, 1).alias("Obesity_%"),
                F.round(F.mean("pa_any")        * 100, 1).alias("Any_PA_%"),
            ).orderBy("age_group"),
            "4.10 Outcomes by Age Group"
        )


# =============================================================================
# SECTION 5: MODELER
# =============================================================================

class BRFSSModeler:
    """
    Inference-focused modelling stage for BRFSS chronic disease analysis.

    Design rationale
    ----------------
    PySpark handles all large-scale data operations in this pipeline
    (3.5 M records across Sections 1–4). For the modelling stage the
    analytic dataset — already cleaned, engineered, and filtered to
    complete cases — is collected to pandas and fitted with statsmodels
    logistic regression.  This mirrors the SAS PROC LOGISTIC pattern
    used by KECC: SAS / PySpark for data processing, a dedicated
    inference engine for coefficient estimation.

    statsmodels provides the full Wald inference table (coefficient,
    standard error, odds ratio, 95 % CI, z-statistic, p-value) that
    is the standard output for epidemiological regression analysis.
    Random Forest runs in PySpark as a supplementary non-parametric
    check on variable importance.

    Outputs
    -------
    Two CSV attachments written to ``config.output_dir``:

    * ``table1_patient_characteristics.csv``  — baseline characteristics
      stratified by diabetes status (N, mean ± SD or %).
    * ``table2_model_estimates.csv``  — logistic regression inference
      table (coeff, SE, OR, 95 % CI, p-value) for every predictor.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session (used for Random Forest stage).
    config : BRFSSConfig
        Pipeline configuration.
    outcome_col : str
        Binary outcome column (default: ``flag_diabetes``).
    """

    # ── Features for statsmodels inference model ────────────────────────────
    # Only variables available in ALL 8 survey years (2017–2024).
    # flag_htn (odd years only) and sleep_hrs/flag_poor_sleep (4 of 8 years)
    # are excluded here — including rotating-core variables causes listwise
    # deletion to remove ~99% of records, leaving an underpowered and
    # potentially singular design matrix.
    # Both variables remain in the Random Forest and EDA (they are valuable
    # there), but are not suitable for a complete-case inference model.
    INFERENCE_FEATURE_COLS: List[str] = [
        # Lifestyle — modifiable (all years)
        "log_pa_min", "meets_pa_guidelines", "pa_any",
        "lifestyle_score",
        # Body composition (all years)
        "bmi", "flag_obese",
        # Demographics (all years)
        "_AGE80", "sex",
        # Mental health (all years)
        "mental_hlth_days", "flag_mental_burden",
        # Clinical comorbidities (all years — CHD available all years)
        "flag_chd",
        # Self-rated health (all years)
        "gen_health",
        # Survey year trend
        "years_since_2017",
    ]

    # ── Features for Random Forest (PySpark) ────────────────────────────────
    # Includes rotating-core variables; RF handles missing via handleInvalid.
    FEATURE_COLS: List[str] = [
        "log_pa_min", "meets_pa_guidelines", "pa_any",
        "sleep_hrs", "flag_poor_sleep",
        "lifestyle_score",
        "bmi", "flag_obese",
        "_AGE80", "sex",
        "mental_hlth_days", "flag_mental_burden",
        "flag_htn", "flag_chd",
        "gen_health",
        "years_since_2017",
    ]

    # Human-readable labels for Table 1 and Table 2 output
    FEATURE_LABELS: Dict[str, str] = {
        "log_pa_min":          "PA volume (log MET-min/wk)",
        "meets_pa_guidelines": "Meets CDC PA guidelines",
        "pa_any":              "Any physical activity",
        "sleep_hrs":           "Sleep duration (hrs/night)",
        "flag_poor_sleep":     "Poor sleep (<7 hrs)",
        "lifestyle_score":     "Healthy lifestyle score (0–4)",
        "bmi":                 "BMI (kg/m²)",
        "flag_obese":          "Obesity (BMI ≥30)",
        "_AGE80":              "Age (years)",
        "sex":                 "Female sex",
        "mental_hlth_days":    "Poor mental health days",
        "flag_mental_burden":  "Mental health burden (≥14 days)",
        "flag_htn":            "Hypertension",
        "flag_chd":            "Coronary heart disease",
        "gen_health":          "Self-rated health (1=Poor–5=Excellent)",
        "years_since_2017":    "Survey year (years since 2017)",
    }

    def __init__(
        self,
        spark: SparkSession,
        config: BRFSSConfig,
        outcome_col: str = "flag_diabetes",
    ) -> None:
        self.spark       = spark
        self.config      = config
        self.outcome_col = outcome_col
        self.logger      = logging.getLogger("brfss.modeler")

    # ──────────────────────────────────────────────────────────────────────────
    # Table 1: Patient characteristics
    # ──────────────────────────────────────────────────────────────────────────

    def build_patient_characteristics(
        self, analytic_df: DataFrame
    ) -> pd.DataFrame:
        """
        Compute Table 1: baseline characteristics stratified by diabetes status.

        For continuous variables the table reports the unweighted mean.
        For binary variables it reports the proportion (%) of positive cases.
        Missing values are excluded from each variable's calculation
        independently, so the denominator may vary by row.

        Parameters
        ----------
        analytic_df : DataFrame
            Full engineered BRFSS Spark DataFrame (all years).

        Returns
        -------
        pd.DataFrame
            Tidy table with columns: Variable, Diabetic, No_Diabetes, Overall.
            Saved to ``table1_patient_characteristics.csv``.
        """
        self.logger.info("Building Table 1: patient characteristics ...")

        groups = {
            "Diabetic":     analytic_df.filter(F.col(self.outcome_col) == 1),
            "No_Diabetes":  analytic_df.filter(F.col(self.outcome_col) == 0),
            "Overall":      analytic_df,
        }
        n_counts = {k: v.count() for k, v in groups.items()}

        self.logger.info(
            "  N — Diabetic: %s | No Diabetes: %s | Overall: %s",
            f"{n_counts['Diabetic']:,}",
            f"{n_counts['No_Diabetes']:,}",
            f"{n_counts['Overall']:,}",
        )

        # Variables to summarise — (column, label, type)
        continuous_vars = [
            ("_AGE80",           "Age (years)"),
            ("bmi",              "BMI (kg/m²)"),
            ("pa_min_week",      "PA (MET-min/week)"),
            ("sleep_hrs",        "Sleep duration (hrs/night)"),
            ("mental_hlth_days", "Poor mental health days/month"),
            ("gen_health",       "Self-rated health (1=Poor–5=Excellent)"),
        ]
        binary_vars = [
            ("pa_any",              "Any physical activity (%)"),
            ("meets_pa_guidelines", "Meets CDC PA guidelines (%)"),
            ("flag_poor_sleep",     "Poor sleep — <7 hrs (%)"),
            ("flag_obese",          "Obesity — BMI ≥30 (%)"),
            ("flag_htn",            "Hypertension (%) [odd yrs only]"),
            ("flag_chd",            "Coronary heart disease (%)"),
            ("flag_mental_burden",  "Mental health burden ≥14 days (%)"),
        ]
        # sex==2 → Female
        sex_var = ("sex", "Female (%)", 2)

        rows: List[Dict] = []

        # N row
        rows.append({
            "Variable": "N",
            "Diabetic":    f"{n_counts['Diabetic']:,}",
            "No_Diabetes": f"{n_counts['No_Diabetes']:,}",
            "Overall":     f"{n_counts['Overall']:,}",
        })

        # Continuous
        for col, label in continuous_vars:
            if col not in analytic_df.columns:
                continue
            row = {"Variable": label}
            for grp_name, grp_df in groups.items():
                val = grp_df.agg(F.round(F.mean(col), 1)).collect()[0][0]
                row[grp_name] = str(val) if val is not None else "—"
            rows.append(row)

        # Sex
        col, label, pos_val = sex_var
        if col in analytic_df.columns:
            row = {"Variable": label}
            for grp_name, grp_df in groups.items():
                n_grp = n_counts[grp_name]
                n_pos = grp_df.filter(F.col(col) == pos_val).count()
                row[grp_name] = f"{round(n_pos/n_grp*100, 1)}%" if n_grp > 0 else "—"
            rows.append(row)

        # Binary
        for col, label in binary_vars:
            if col not in analytic_df.columns:
                continue
            row = {"Variable": label}
            for grp_name, grp_df in groups.items():
                n_grp = n_counts[grp_name]
                n_pos = grp_df.filter(F.col(col) == 1).count()
                row[grp_name] = f"{round(n_pos/n_grp*100, 1)}%" if n_grp > 0 else "—"
            rows.append(row)

        table1 = pd.DataFrame(rows, columns=["Variable", "Diabetic", "No_Diabetes", "Overall"])

        # Log to console in formatted layout
        self.logger.info("=" * 60)
        self.logger.info("TABLE 1: Patient Characteristics by Diabetes Status")
        self.logger.info("=" * 60)
        self.logger.info(
            "  %-40s %12s %14s %12s",
            "Variable", "Diabetic", "No Diabetes", "Overall"
        )
        self.logger.info("  " + "-" * 80)
        for _, r in table1.iterrows():
            self.logger.info(
                "  %-40s %12s %14s %12s",
                r["Variable"], r["Diabetic"], r["No_Diabetes"], r["Overall"]
            )
        self.logger.info(
            "  Note: HTN available in odd survey years only (rotating core)."
        )

        return table1

    # ──────────────────────────────────────────────────────────────────────────
    # Inference: statsmodels logistic regression
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_modeling_data(self, analytic_df: DataFrame) -> pd.DataFrame:
        """
        Collect the modeling subset from Spark to pandas for inference.

        Selects the outcome and all available feature columns, drops rows
        with missing values in the required core variables (outcome, age,
        BMI, sex), and returns a pandas DataFrame ready for statsmodels.

        PySpark handles the filtering and projection at scale before the
        collect — only the modeling-eligible subset is transferred to the
        driver.

        Parameters
        ----------
        analytic_df : DataFrame

        Returns
        -------
        pd.DataFrame
            Complete-case dataset for logistic regression.
        """
        available_features = [
            c for c in self.INFERENCE_FEATURE_COLS if c in analytic_df.columns
        ]
        keep_cols = [self.outcome_col] + available_features
        # Require non-null on outcome and all inference features — these are
        # all available in every survey year so listwise deletion is minimal.
        required  = [self.outcome_col] + available_features

        modeling_sdf = (
            analytic_df
            .select(keep_cols)
            .dropna(subset=required)
        )
        n_total = modeling_sdf.count()
        self.logger.info(
            "Modeling dataset: %s records after core-variable missingness filter",
            f"{n_total:,}"
        )

        self.logger.info(
            "Collecting modeling dataset to pandas for statsmodels inference ..."
        )
        modeling_pd = modeling_sdf.toPandas()
        self.logger.info(
            "  Collected: %s rows × %s cols", f"{len(modeling_pd):,}", len(modeling_pd.columns)
        )
        return modeling_pd, available_features

    def fit_logistic_regression(
        self, analytic_df: DataFrame
    ) -> Tuple[object, pd.DataFrame, List[str]]:
        """
        Fit a logistic regression model using statsmodels and produce
        the full Wald inference table.

        statsmodels Logit is used rather than PySpark LogisticRegression
        because it provides the complete Wald inference output —
        coefficient, standard error, z-statistic, p-value, and 95 % CI —
        which is the standard reporting format for epidemiological
        regression analysis (equivalent to SAS PROC LOGISTIC output).

        PySpark is used upstream for all large-scale data operations;
        statsmodels is the inference engine on the collected modeling subset.

        Parameters
        ----------
        analytic_df : DataFrame
            Full engineered BRFSS Spark DataFrame.

        Returns
        -------
        result : statsmodels.discrete.discrete_model.BinaryResultsWrapper
            Fitted model object.
        table2 : pd.DataFrame
            Table 2 — coefficient estimates with OR, CI, and p-values.
        feature_names : List[str]
            Names of features in model order.
        """
        modeling_pd, feature_names = self._collect_modeling_data(analytic_df)

        # All rows are already complete-case (dropna applied in _collect_modeling_data)
        self.logger.info(
            "Complete-case N for logistic regression: %s", f"{len(modeling_pd):,}"
        )

        y = modeling_pd[self.outcome_col].astype(float)
        X = modeling_pd[feature_names].astype(float)
        X = sm.add_constant(X)   # adds intercept column 'const'

        self.logger.info("Fitting statsmodels Logit (MLE, no regularization) ...")
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(
            maxiter=200,
            disp=False,        # suppress iteration output
            method="newton",   # Newton-Raphson, same algorithm as SAS PROC LOGISTIC
        )
        self.logger.info(
            "  Converged: %s | Log-likelihood: %.2f | Pseudo-R²: %.4f",
            result.mle_retvals.get("converged", "?"),
            result.llf,
            result.prsquared,
        )

        # ── Build Table 2 ─────────────────────────────────────────────────────
        coefs   = result.params
        ses     = result.bse
        zstats  = result.tvalues
        pvals   = result.pvalues
        ci      = result.conf_int(alpha=0.05)   # 95 % Wald CI on log-odds scale
        OR      = np.exp(coefs)
        ci_or   = np.exp(ci)

        rows = []
        for var in coefs.index:
            label = self.FEATURE_LABELS.get(var, var)
            sig   = (
                "***" if pvals[var] < 0.001 else
                "**"  if pvals[var] < 0.01  else
                "*"   if pvals[var] < 0.05  else ""
            )
            rows.append({
                "Variable":      var,
                "Label":         label,
                "Coefficient":   round(coefs[var], 4),
                "SE":            round(ses[var], 4),
                "OR":            round(OR[var], 4),
                "CI_lower":      round(ci_or.loc[var, 0], 4),
                "CI_upper":      round(ci_or.loc[var, 1], 4),
                "z":             round(zstats[var], 3),
                "p_value":       round(pvals[var], 4),
                "Significance":  sig,
            })

        table2 = pd.DataFrame(rows)

        # ── Log Table 2 to console ────────────────────────────────────────────
        self.logger.info("=" * 60)
        self.logger.info(
            "TABLE 2: Logistic Regression Estimates — outcome: %s",
            self.outcome_col
        )
        self.logger.info("=" * 60)
        self.logger.info(
            "  %-32s %8s %8s %8s %18s %10s",
            "Variable", "Coeff", "SE", "OR", "95% CI", "p-value"
        )
        self.logger.info("  " + "-" * 88)

        for _, r in table2.iterrows():
            ci_str = f"({r['CI_lower']:.3f}, {r['CI_upper']:.3f})"
            self.logger.info(
                "  %-32s %8.4f %8.4f %8.4f %18s %10s %s",
                r["Label"], r["Coefficient"], r["SE"],
                r["OR"], ci_str, f"{r['p_value']:.4f}", r["Significance"]
            )
            # Interpretation line — mirrors SAS tutorial style
            if r["Variable"] != "const":
                direction  = "increases" if r["Coefficient"] > 0 else "decreases"
                pct_change = abs(r["OR"] - 1) * 100
                self.logger.info(
                    "    → 1-unit ↑ in %-28s %s diabetes odds by %.1f%% "
                    "(OR=%.3f, 95%%CI: %.3f–%.3f, p=%s)",
                    r["Label"], direction, pct_change,
                    r["OR"], r["CI_lower"], r["CI_upper"],
                    "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
                )

        self.logger.info("  " + "-" * 88)
        self.logger.info(
            "  OR = odds ratio; CI = 95%% Wald confidence interval; "
            "*** p<0.001  ** p<0.01  * p<0.05"
        )
        self.logger.info(
            "  Model: statsmodels Logit, Newton-Raphson MLE, %s observations",
            f"{len(modeling_pd):,}"
        )

        return result, table2, feature_names

    # ──────────────────────────────────────────────────────────────────────────
    # Supplementary: Random Forest (PySpark) — feature importance
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(self, preds: DataFrame, model_name: str) -> Dict[str, float]:
        """
        Compute AUC-ROC and accuracy on PySpark prediction DataFrame.

        Parameters
        ----------
        preds : DataFrame
            Spark DataFrame with ``probability`` and ``prediction`` columns.
        model_name : str
            Label for logging.

        Returns
        -------
        Dict[str, float]
            ``{'auc': float, 'accuracy': float}``
        """
        auc = BinaryClassificationEvaluator(
            labelCol=self.outcome_col, metricName="areaUnderROC"
        ).evaluate(preds)
        acc = MulticlassClassificationEvaluator(
            labelCol=self.outcome_col,
            predictionCol="prediction",
            metricName="accuracy",
        ).evaluate(preds)
        self.logger.info(
            "%s — AUC: %.4f | Accuracy: %.4f", model_name, auc, acc
        )
        return {"auc": auc, "accuracy": acc}

    def fit_random_forest(
        self, analytic_df: DataFrame
    ) -> Dict[str, float]:
        """
        Fit a Random Forest classifier in PySpark (supplementary).

        Provides a non-parametric, non-linear benchmark for comparison with
        the logistic regression, and outputs ranked feature importances to
        corroborate the inferential findings from Table 2.

        Parameters
        ----------
        analytic_df : DataFrame
            Full engineered BRFSS Spark DataFrame.

        Returns
        -------
        Dict[str, float]
            AUC and accuracy on the held-out test set.
        """
        self.logger.info("Fitting Random Forest (supplementary, PySpark) ...")

        available_features = [
            c for c in self.FEATURE_COLS if c in analytic_df.columns
        ]
        required = [self.outcome_col, "bmi", "_AGE80", "sex"]
        modeling_sdf = analytic_df.select(
            [self.outcome_col] + available_features
        ).dropna(subset=required)

        train_sdf, test_sdf = modeling_sdf.randomSplit(
            [self.config.train_ratio, 1 - self.config.train_ratio],
            seed=self.config.seed,
        )
        self.logger.info(
            "  RF train N = %s | test N = %s",
            f"{train_sdf.count():,}", f"{test_sdf.count():,}"
        )

        assembler = VectorAssembler(
            inputCols=available_features,
            outputCol="features_raw",
            handleInvalid="skip",
        )
        scaler = StandardScaler(
            inputCol="features_raw", outputCol="features",
            withMean=True, withStd=True,
        )
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=self.outcome_col,
            numTrees=50,
            maxDepth=6,
            seed=self.config.seed,
        )
        pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
        model_rf    = pipeline_rf.fit(train_sdf)
        preds_rf    = model_rf.transform(test_sdf)
        metrics_rf  = self._evaluate(preds_rf, "Random Forest")

        # Feature importances
        rf_fitted   = model_rf.stages[-1]
        importances = rf_fitted.featureImportances.toArray()
        feat_imp    = sorted(
            zip(available_features, importances),
            key=lambda x: x[1], reverse=True
        )
        self.logger.info("RF Feature Importances (top 10):")
        for rank, (feat, imp) in enumerate(feat_imp[:10], 1):
            label = self.FEATURE_LABELS.get(feat, feat)
            bar   = "█" * int(imp * 200)
            self.logger.info(
                "  %2d. %-38s %.4f  %s", rank, label, imp, bar
            )

        return metrics_rf

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run_analysis(
        self, analytic_df: DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Execute the full modelling and inference stage.

        Steps
        -----
        1. Build Table 1 (patient characteristics) using PySpark aggregations.
        2. Collect modeling subset to pandas and fit statsmodels Logit for
           full Wald inference — produces Table 2.
        3. Fit Random Forest in PySpark for supplementary feature importance.

        Parameters
        ----------
        analytic_df : DataFrame
            Full engineered BRFSS Spark DataFrame from BRFSSEngineer.

        Returns
        -------
        table1 : pd.DataFrame
        table2 : pd.DataFrame
        rf_metrics : Dict[str, float]
        """
        self.logger.info("=" * 60)
        self.logger.info(
            "SECTION 5: Modelling & Inference — outcome: %s",
            self.outcome_col,
        )
        self.logger.info("=" * 60)

        # Step 1 — Table 1
        table1 = self.build_patient_characteristics(analytic_df)

        # Step 2 — Logistic regression inference (statsmodels)
        _, table2, _ = self.fit_logistic_regression(analytic_df)

        # Step 3 — Random Forest (PySpark, supplementary)
        rf_metrics = self.fit_random_forest(analytic_df)

        return table1, table2, rf_metrics




# =============================================================================
# SECTION 6: PIPELINE ORCHESTRATOR
# =============================================================================

class BRFSSPipeline:
    """
    End-to-end orchestrator for the BRFSS analysis pipeline.

    Coordinates BRFSSLoader → BRFSSCleaner → BRFSSEngineer →
    BRFSSAnalyzer → BRFSSModeler, caching intermediate DataFrames
    where appropriate and saving the final scored dataset as Parquet.

    Parameters
    ----------
    config : BRFSSConfig
        Central configuration. Defaults to BRFSSConfig() which reads
        from ./data/ and writes to ./brfss_output/.
    """

    def __init__(self, config: Optional[BRFSSConfig] = None) -> None:
        self.config = config or BRFSSConfig()
        self.logger = logging.getLogger("brfss.pipeline")
        self.spark: Optional[SparkSession] = None

    def _create_spark(self) -> SparkSession:
        """
        Create and return a configured SparkSession.

        Returns
        -------
        SparkSession
        """
        self.logger.info("Initializing SparkSession ...")
        spark = (
            SparkSession.builder
            .appName("BRFSS_2017_2024_Analysis")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", "6g")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        self.logger.info("SparkSession ready.")
        return spark

    def execute_pipeline(self) -> None:
        """
        Execute the full BRFSS pipeline from data ingestion to model output.

        Pipeline stages:
        1. Ingest: load all XPT files → combined Spark DataFrame.
        2. Clean:  replace missing codes, cast types, apply business rules.
        3. Engineer: derive outcome flags, composite scores, log transforms.
        4. Analyze: EDA — trends, group comparisons, prevalence tables.
        5. Model:  logistic regression + random forest for diabetes risk.
        6. Export: save scored dataset as Parquet.
        """
        start = time.time()
        self.logger.info("=" * 60)
        self.logger.info("BRFSS 2017–2024 Pipeline  |  Yuntao (Kevin) Tan")
        self.logger.info("=" * 60)

        # ── Stage 0: Spark ────────────────────────────────────────────────────
        self.spark = self._create_spark()

        # ── Stage 1: Load ─────────────────────────────────────────────────────
        loader = BRFSSLoader(self.spark, self.config)
        df_raw = loader.load_all_years()
        df_raw.cache()

        # ── Stage 2: Clean ────────────────────────────────────────────────────
        cleaner = BRFSSCleaner(self.config)
        df_clean = cleaner.standardize_survey_data(df_raw)
        df_raw.unpersist()
        df_clean.cache()

        # ── Stage 3: Feature engineering ──────────────────────────────────────
        engineer = BRFSSEngineer(self.config)
        df_analytic = engineer.build_analytic_features(df_clean)
        df_clean.unpersist()
        df_analytic.cache()

        total = df_analytic.count()
        self.logger.info("Analytic dataset ready: %d records", total)

        # ── Stage 4: EDA ──────────────────────────────────────────────────────
        analyzer = BRFSSAnalyzer(self.config)
        analyzer.compute_descriptive_statistics(df_analytic)

        # ── Stage 5: Modelling & Inference ───────────────────────────────────
        modeler = BRFSSModeler(self.spark, self.config, outcome_col="flag_diabetes")
        table1, table2, rf_metrics = modeler.run_analysis(df_analytic)

        # ── Stage 6: Export ───────────────────────────────────────────────────
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Scored Spark dataset
        output_path = os.path.join(self.config.output_dir, "brfss_scored")
        self.logger.info("Saving scored dataset to: %s", output_path)
        export_cols = [
            "survey_year", "_STATE", "sex_label", "age_group", "bmi_cat",
            "bmi", "sleep_hrs", "pa_min_week", "meets_pa_guidelines",
            "mental_hlth_days", "gen_health", "lifestyle_score",
            "disease_burden",
            "flag_diabetes", "flag_htn", "flag_ckd", "flag_chd",
            "flag_obese", "flag_poor_sleep", "flag_mental_burden",
            "_LLCPWT",
        ]
        available_export = [c for c in export_cols if c in df_analytic.columns]
        df_analytic.select(available_export).write.mode("overwrite").parquet(output_path)

        # Attachment 1 — Table 1: patient characteristics
        t1_path = os.path.join(self.config.output_dir, "table1_patient_characteristics.csv")
        table1.to_csv(t1_path, index=False)
        self.logger.info("Attachment 1 saved: %s", t1_path)

        # Attachment 2 — Table 2: logistic regression estimates
        t2_path = os.path.join(self.config.output_dir, "table2_model_estimates.csv")
        table2.to_csv(t2_path, index=False)
        self.logger.info("Attachment 2 saved: %s", t2_path)

        # ── Summary ───────────────────────────────────────────────────────────
        elapsed = time.time() - start
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
        self.logger.info("  Total records  : %d", total)
        self.logger.info("  RF  AUC        : %.4f", rf_metrics["auc"])
        self.logger.info("  Output path    : %s", output_path)
        self.logger.info("  Table 1 (CSV)  : %s", t1_path)
        self.logger.info("  Table 2 (CSV)  : %s", t2_path)
        self.logger.info("=" * 60)

        df_analytic.unpersist()
        self.spark.stop()


# =============================================================================
# SECTION 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    config = BRFSSConfig(
        data_dir="./data",       # folder containing LLCP2017.XPT ~ LLCP2024.XPT
        output_dir="./brfss_output",
        years=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    )
    pipeline = BRFSSPipeline(config=config)
    pipeline.execute_pipeline()
