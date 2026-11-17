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


def build_model() -> Pipeline:
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("categorical", categorical, CATEGORICAL),
        ("numeric", numeric, NUMERIC),
    ])
    return Pipeline([
        ("features", prep),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def analyze(input_path: Path, output_dir: Path) -> None:
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
    observed, predicted = calibration_curve(test["whiff"], probabilities, n_bins=10, strategy="quantile")

    output_dir.mkdir(parents=True, exist_ok=True)
    rates = (frame.groupby(["pitch_type", "count"], observed=True)
             .agg(pitches=("whiff", "size"), whiffs=("whiff", "sum"))
             .reset_index())
    rates["whiff_rate"] = rates["whiffs"] / rates["pitches"]
    rates.to_csv(output_dir / "whiff_rates.csv", index=False)

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
        f"- Holdout ROC AUC: {auc:.3f}\n"
        f"- Holdout Brier score: {brier:.4f}\n\n"
        "The target is whiff among recorded swing outcomes. These results are predictive and "
        "descriptive; they do not estimate the causal effect of changing pitch usage. Consult "
        "`whiff_rates.csv` for pitch-count sample sizes and `calibration.png` for probability "
        "calibration.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/statcast.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()
