"""Analyze pitch-level whiff outcomes and evaluate a temporal baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve

REQUIRED = {"game_date", "description", "pitch_type", "balls", "strikes", "stand", "p_throws"}
CATEGORICAL = ["pitch_type", "count", "stand", "p_throws"]
NUMERIC = ["release_speed"]
RATE_GROUPS = ["pitch_type", "count", "stand", "p_throws"]


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and create the modeling rows and target."""
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing required Statcast columns: {', '.join(sorted(missing))}")
    frame = data.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["description"] = frame["description"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["game_date", "pitch_type", "balls", "strikes", "stand", "p_throws"])
    frame["count"] = frame["balls"].astype(int).astype(str) + "-" + frame["strikes"].astype(int).astype(str)
    for base in ("1b", "2b", "3b"):
        runner_col = f"on_{base}"
        frame[f"runner_on_{base}"] = frame[runner_col].notna().astype(int) if runner_col in frame else 0
    # The feed uses values such as " swinging_strike"; strip whitespace before matching.
    frame["whiff"] = frame["description"].str.lower().isin(
        {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
    ).astype(int)
    # Exclude takes and foul balls: the target compares misses with contact/in-play swings.
    swing_outcomes = {
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
        "foul_bunt", "missed_bunt", "hit_into_play", "hit_into_play_no_out",
        "hit_into_play_score",
    }
    frame = frame[frame["description"].str.lower().isin(swing_outcomes)].copy()
    if "release_speed" in frame:
        frame["release_speed"] = pd.to_numeric(frame["release_speed"], errors="coerce")
    else:
        frame["release_speed"] = np.nan
    frame = frame.dropna(subset=["pitch_type", "stand", "p_throws"])
    frame = frame[frame["pitch_type"].astype(str).str.len() > 0]
    return frame.sort_values("game_date").reset_index(drop=True)


def build_model(categorical: list[str] | None = None,
                numeric: list[str] | None = None) -> Pipeline:
    categorical_columns = CATEGORICAL if categorical is None else categorical
    numeric_columns = NUMERIC if numeric is None else numeric
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = [("categorical", categorical_pipeline, categorical_columns)]
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    prep = ColumnTransformer(transformers)
    return Pipeline([
        ("features", prep),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def wilson_interval(successes: pd.Series, total: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """Return two-sided Wilson score bounds for binomial proportions."""
    rate = successes / total
    z2 = z**2
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    margin = z * np.sqrt((rate * (1 - rate) / total) + (z2 / (4 * total**2))) / denominator
    return (center - margin).clip(lower=0), (center + margin).clip(upper=1)


def analyze(input_path: Path, output_dir: Path, min_pitches: int = 50) -> None:
    if min_pitches < 1:
        raise ValueError("min_pitches must be at least 1.")
    raw = pd.read_csv(input_path, low_memory=False)
    frame = prepare(raw)
    if frame["whiff"].nunique() != 2 or len(frame) < 100:
        raise ValueError("Need at least 100 swing outcomes and both whiffs and contact to analyze.")

    split_at = frame["game_date"].quantile(0.8)
    train = frame[frame["game_date"] < split_at]
    test = frame[frame["game_date"] >= split_at]
    if train.empty or test.empty or train["whiff"].nunique() != 2 or test["whiff"].nunique() != 2:
        raise ValueError("The chronological split needs both outcome classes in train and test.")

    model = build_model()
    model.fit(train[CATEGORICAL + NUMERIC], train["whiff"])
    probabilities = model.predict_proba(test[CATEGORICAL + NUMERIC])[:, 1]
    auc = roc_auc_score(test["whiff"], probabilities)
    brier = brier_score_loss(test["whiff"], probabilities)
    baseline_probability = train["whiff"].mean()
    baseline_brier = brier_score_loss(test["whiff"], np.full(len(test), baseline_probability))
    observed, predicted = calibration_curve(test["whiff"], probabilities, n_bins=10, strategy="quantile")

    output_dir.mkdir(parents=True, exist_ok=True)
    rates = (frame.groupby(RATE_GROUPS, observed=True)
             .agg(pitches=("whiff", "size"), whiffs=("whiff", "sum"))
             .reset_index())
    rates["whiff_rate"] = rates["whiffs"] / rates["pitches"]
    rates["ci_95_low"], rates["ci_95_high"] = wilson_interval(rates["whiffs"], rates["pitches"])
    rates["meets_min_pitches"] = rates["pitches"] >= min_pitches
    rates.to_csv(output_dir / "whiff_rates.csv", index=False)
    qualifying_groups = int(rates["meets_min_pitches"].sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", label="Temporal holdout")
    ax.set(xlabel="Mean predicted whiff probability", ylabel="Observed whiff rate",
           title="Whiff model calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=160)
    plt.close(fig)

    summary = (
        "# Pitch Mix Lab: analysis summary\n\n"
        f"- Modeling rows: {len(frame):,}\n"
        f"- Train rows: {len(train):,}; holdout rows: {len(test):,}\n"
        f"- Temporal split date: {split_at.date()}\n"
        f"- Train whiff rate: {baseline_probability:.3f}; holdout whiff rate: {test['whiff'].mean():.3f}\n"
        f"- Holdout ROC AUC: {auc:.3f}\n"
        f"- Holdout Brier score: {brier:.4f}; constant-rate baseline Brier score: {baseline_brier:.4f}\n\n"
        f"- Rate table groups with at least {min_pitches} pitches: {qualifying_groups:,} of {len(rates):,}\n\n"
        "The target is whiff among recorded swing outcomes. These results are predictive and "
        "descriptive; they do not estimate the causal effect of changing pitch usage. Consult "
        "`whiff_rates.csv` for pitch-count sample sizes, 95% Wilson intervals, and batter/pitcher "
        "handedness splits. `meets_min_pitches` flags groups large enough for the chosen threshold; "
        "it does not establish that an observed difference is meaningful or causal. Consult "
        "`calibration.png` for probability calibration.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/statcast.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--min-pitches", type=int, default=50,
                        help="Flag rate-table groups with fewer pitches as below this sample threshold (default: 50).")
    args = parser.parse_args()
    analyze(args.input, args.output, args.min_pitches)


if __name__ == "__main__":
    main()
