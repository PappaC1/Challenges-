"""
South African Horse Racing Prediction Model
===========================================

This module:

1. Loads and validates historical race results and today's runners.
2. Creates racing features.
3. Trains a LightGBM classifier to estimate each runner's probability of winning.
4. Produces three selections per race:
   - Probable Winner
   - Value Bet
   - Danger Horse

It is intended for research and decision support only.
Horse racing is uncertain and no model can guarantee profitable selections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Required CSV columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "race_id",
    "race_date",
    "course",
    "race_number",
    "distance_m",
    "going",
    "class",
    "race_type",
    "horse_name",
    "horse_id",
    "trainer_name",
    "jockey_name",
    "draw",
    "weight_kg",
    "age",
    "sex",
    "days_since_last_run",
    "runs_last_90_days",
    "wins_last_12_months",
    "places_last_12_months",
    "starts_last_12_months",
    "last_3_positions",
    "last_win_days_ago",
    "official_rating",
    "market_odds",
    "finished_position",
]


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file and ensure that all required racing columns are present.

    Parameters
    ----------
    path : str
        Path to a historical-results CSV or today's-runners CSV.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The file '{path}' is missing these required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(f"The file '{path}' contains no rows.")

    return df


def parse_last_3_positions(value) -> list[int]:
    """
    Convert a value such as '1,4,2' into.

    Missing, non-numeric, or unplaced results are treated as 99.
    """
    if pd.isna(value) or str(value).strip() == "":
        return [99, 99, 99]

    raw_positions = str(value).replace(" ", "").split(",")
    parsed_positions: list[int] = []

    for raw_position in raw_positions[:3]:
        try:
            parsed_positions.append(int(float(raw_position)))
        except ValueError:
            parsed_positions.append(99)

    while len(parsed_positions) < 3:
        parsed_positions.append(99)

    return parsed_positions


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw racing data and create model features.

    Parameters
    ----------
    df : pd.DataFrame
        Historical results or today's runners.

    Returns
    -------
    pd.DataFrame
        DataFrame containing original and engineered features.
    """
    df = df.copy()

    # Convert core numeric fields
    numeric_columns = [
        "race_number",
        "distance_m",
        "draw",
        "weight_kg",
        "age",
        "days_since_last_run",
        "runs_last_90_days",
        "wins_last_12_months",
        "places_last_12_months",
        "starts_last_12_months",
        "last_win_days_ago",
        "official_rating",
        "market_odds",
        "finished_position",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # Recent form
    positions = df["last_3_positions"].apply(parse_last_3_positions)

    df["last_finish"] = [positions[0] for positions in positions]
    df["second_last_finish"] = [positions[1] for positions in positions]
    df["third_last_finish"] = [positions[2] for positions in positions]

    df["recent_form_score"] = (
        (4 - df["last_finish"].clip(upper=4)) * 3
        + (4 - df["second_last_finish"].clip(upper=4)) * 2
        + (4 - df["third_last_finish"].clip(upper=4))
    )

    # Strike rates
    df["win_rate_12m"] = df["wins_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)

    df["place_rate_12m"] = df["places_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)

    # Market-derived features
    df["market_odds"] = df["market_odds"].clip(lower=1.01)
    df["implied_probability"] = 1.0 / df["market_odds"]
    df["log_odds"] = np.log(df["market_odds"])

    df["market_rank"] = df.groupby("race_id")["market_odds"].rank(
        method="min"
    )

    df["field_size"] = df.groupby("race_id")["horse_name"].transform("count")

    # Historical course and course-distance performance
    historical = df[df["finished_position"].notna()].copy()
    historical["won"] = (historical["finished_position"] == 1).astype(int)

    if not historical.empty:
        global_win_rate = float(historical["won"].mean())

        course_distance_win_rate = (
            historical.groupby(["course", "distance_m"])["won"]
            .mean()
            .rename("course_distance_win_rate")
        )

        course_win_rate = (
            historical.groupby("course")["won"]
            .mean()
            .rename("course_win_rate")
        )

        trainer_win_rate = (
            historical.groupby("trainer_name")["won"]
            .mean()
            .rename("trainer_win_rate")
        )

        jockey_win_rate = (
            historical.groupby("jockey_name")["won"]
            .mean()
            .rename("jockey_win_rate")
        )

        df = df.merge(
            course_distance_win_rate,
            on=["course", "distance_m"],
            how="left",
        )

        df = df.merge(course_win_rate, on="course", how="left")
        df = df.merge(trainer_win_rate, on="trainer_name", how="left")
        df = df.merge(jockey_win_rate, on="jockey_name", how="left")

        df["course_distance_win_rate"] = df[
            "course_distance_win_rate"
        ].fillna(global_win_rate)

        df["course_win_rate"] = df["course_win_rate"].fillna(global_win_rate)
        df["trainer_win_rate"] = df["trainer_win_rate"].fillna(global_win_rate)
        df["jockey_win_rate"] = df["jockey_win_rate"].fillna(global_win_rate)

    else:
        default_rate = 0.12

        df["course_distance_win_rate"] = default_rate
        df["course_win_rate"] = default_rate
        df["trainer_win_rate"] = default_rate
        df["jockey_win_rate"] = default_rate

    # Fill remaining missing values in derived numeric columns
    derived_numeric_columns = [
        "win_rate_12m",
        "place_rate_12m",
        "implied_probability",
        "log_odds",
        "market_rank",
        "field_size",
    ]

    for column in derived_numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "distance_m",
    "draw",
    "weight_kg",
    "age",
    "days_since_last_run",
    "runs_last_90_days",
    "wins_last_12_months",
    "places_last_12_months",
    "starts_last_12_months",
    "last_finish",
    "second_last_finish",
    "third_last_finish",
    "recent_form_score",
    "win_rate_12m",
    "place_rate_12m",
    "log_odds",
    "implied_probability",
    "market_rank",
    "field_size",
    "course_distance_win_rate",
    "course_win_rate",
    "trainer_win_rate",
    "jockey_win_rate",
    "last_win_days_ago",
    "official_rating",
]

CATEGORICAL_FEATURES = [
    "course",
    "going",
    "class",
    "race_type",
    "sex",
    "trainer_name",
    "jockey_name",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def make_model(model_type: str = "lightgbm"):
    """
    Create a scikit-learn pipeline containing preprocessing and a classifier.

    Parameters
    ----------
    model_type : str
        Either "lightgbm" or "logistic".

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", "passthrough"),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),

