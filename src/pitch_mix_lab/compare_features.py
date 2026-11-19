"""Compare feature blocks with expanding-window, out-of-time validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from pitch_mix_lab.analyze import build_model, prepare

BASE_CATEGORICAL = ["pitch_type", "count", "stand", "p_throws"]
GAME_STATE_CATEGORICAL = ["inning_topbot"]
GAME_STATE_NUMERIC = [
    "outs_when_up", "inning", "bat_score_diff", "runner_on_1b", "runner_on_2b", "runner_on_3b"
]
PLAYER_ID_CATEGORICAL = ["pitcher", "batter"]
POST_RELEASE_NUMERIC = [
    "release_speed", "plate_x", "plate_z", "pfx_x", "pfx_z", "release_spin_rate", "release_extension"
]


def feature_blocks() -> list[dict[str, object]]:
    """Return independent candidate additions to the same pre-pitch baseline."""
    return [
        {
            "feature_set": "pre_pitch_baseline",
            "added_block": "None",
            "availability": "Candidate pre-pitch information",
            "categorical": BASE_CATEGORICAL,
            "numeric": [],
        },
        {
            "feature_set": "baseline_plus_release_speed",
            "added_block": "Release speed",
            "availability": "Measured at delivery; not available before the pitch",
            "categorical": BASE_CATEGORICAL,
            "numeric": ["release_speed"],
        },
        {
            "feature_set": "baseline_plus_game_state",
            "added_block": "Pre-pitch game state",
            "availability": "Candidate pre-pitch information",
            "categorical": BASE_CATEGORICAL + GAME_STATE_CATEGORICAL,
            "numeric": GAME_STATE_NUMERIC,
        },
        {
            "feature_set": "baseline_plus_player_ids",
            "added_block": "Pitcher and batter IDs",
            "availability": "Known before the pitch; may memorize recurring players",
            "categorical": BASE_CATEGORICAL + PLAYER_ID_CATEGORICAL,
            "numeric": [],
        },
        {
            "feature_set": "baseline_plus_observed_pitch_tracking",
            "added_block": "Speed, location, movement, spin, and extension",
            "availability": "Measured during or after pitch delivery; diagnostic only",
            "categorical": BASE_CATEGORICAL,
            "numeric": POST_RELEASE_NUMERIC,
        },
    ]


def rolling_date_splits(development: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create three equal-calendar validation windows with expanding training history."""
    start = development["game_date"].min().normalize()
    end = development["game_date"].max().normalize() + pd.Timedelta(days=1)
    span = end - start
    boundaries = [
        start + span * 0.50,
        start + span * (2 / 3),
        start + span * (5 / 6),
        end,
    ]
    splits = [(boundaries[i], boundaries[i + 1]) for i in range(3)]
    for i, (valid_start, valid_end) in enumerate(splits, start=1):
        train = development[development["game_date"] < valid_start]
        valid = development[(development["game_date"] >= valid_start) &
                            (development["game_date"] < valid_end)]
        if train.empty or valid.empty:
            raise ValueError(f"Rolling validation window {i} has no training or validation rows.")
        if train["whiff"].nunique() != 2 or valid["whiff"].nunique() != 2:
            raise ValueError(f"Rolling validation window {i} must contain both target classes.")
    return splits


def compare(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(input_path, low_memory=False)
    frame = prepare(raw)
    if frame["whiff"].nunique() != 2 or len(frame) < 100:
        raise ValueError("Need at least 100 swing outcomes and both whiffs and contact to compare features.")

    # Reserve the same latest-date holdout as the main analysis; feature selection uses earlier data only.
    final_holdout_start = frame["game_date"].quantile(0.8)
    development = frame[frame["game_date"] < final_holdout_start].copy()
    splits = rolling_date_splits(development)
    blocks = feature_blocks()

    required_features = set()
    for block in blocks:
        required_features.update(block["categorical"])
        required_features.update(block["numeric"])
    missing = required_features - set(development.columns)
    if missing:
        raise ValueError(f"Input is missing candidate feature columns: {', '.join(sorted(missing))}")

    records: list[dict[str, object]] = []
    for fold, (valid_start, valid_end) in enumerate(splits, start=1):
        train = development[development["game_date"] < valid_start]
        valid = development[(development["game_date"] >= valid_start) &
                            (development["game_date"] < valid_end)]
        constant_rate = float(train["whiff"].mean())
        constant_brier = brier_score_loss(valid["whiff"], np.full(len(valid), constant_rate))

        for block in blocks:
            categorical = list(block["categorical"])
            numeric = list(block["numeric"])
            model = build_model(categorical=categorical, numeric=numeric)
            predictors = categorical + numeric
            model.fit(train[predictors], train["whiff"])
            probability = model.predict_proba(valid[predictors])[:, 1]
            records.append({
                "fold": fold,
                "train_start": train["game_date"].min().date().isoformat(),
                "train_end": train["game_date"].max().date().isoformat(),
                "validation_start": valid["game_date"].min().date().isoformat(),
                "validation_end": valid["game_date"].max().date().isoformat(),
                "train_rows": len(train),
                "validation_rows": len(valid),
                "feature_set": block["feature_set"],
                "added_block": block["added_block"],
                "availability": block["availability"],
                "categorical_features": ";".join(categorical),
                "numeric_features": ";".join(numeric),
                "roc_auc": roc_auc_score(valid["whiff"], probability),
                "brier_score": brier_score_loss(valid["whiff"], probability),
                "constant_rate_brier": constant_brier,
            })

    fold_scores = pd.DataFrame(records)
    base_by_fold = (fold_scores[fold_scores["feature_set"] == "pre_pitch_baseline"]
                    .set_index("fold")["brier_score"])
    fold_scores["brier_delta_vs_pre_pitch_baseline"] = fold_scores.apply(
        lambda row: row["brier_score"] - base_by_fold.loc[row["fold"]], axis=1
    )

    summary = (fold_scores.groupby(["feature_set", "added_block", "availability"], as_index=False)
               .agg(folds=("fold", "nunique"), mean_roc_auc=("roc_auc", "mean"),
                    sd_roc_auc=("roc_auc", "std"), mean_brier=("brier_score", "mean"),
                    sd_brier=("brier_score", "std"), mean_constant_brier=("constant_rate_brier", "mean"),
                    mean_brier_delta_vs_pre_pitch_baseline=("brier_delta_vs_pre_pitch_baseline", "mean")))

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output_dir / "feature_block_scores.csv", index=False)
    summary.to_csv(output_dir / "feature_block_summary.csv", index=False)

    lines = [
        "# Feature block comparison",
        "",
        f"The latest-date holdout begins {final_holdout_start.date()} and was not used to select features.",
        f"Feature blocks were compared on {len(splits)} expanding-window validation periods drawn from earlier dates.",
        "Each candidate adds one feature block to the same pre-pitch baseline, except the release-speed row, which reproduces the current model's delivery-time feature.",
        "",
        "## Summary",
        "",
        "| Feature set | Availability | ROC AUC mean (SD) | Brier mean (SD) | Brier delta vs pre-pitch baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        auc_sd = row.sd_roc_auc if pd.notna(row.sd_roc_auc) else 0.0
        brier_sd = row.sd_brier if pd.notna(row.sd_brier) else 0.0
        lines.append(
            f"| {row.feature_set} | {row.availability} | {row.mean_roc_auc:.3f} ({auc_sd:.3f}) | "
            f"{row.mean_brier:.4f} ({brier_sd:.4f}) | {row.mean_brier_delta_vs_pre_pitch_baseline:+.4f} |"
        )
    lines.extend([
        "",
        "Lower Brier score is better; higher ROC AUC is better. A negative Brier delta means lower average Brier error than the pre-pitch baseline.",
        "",
        "## Interpretation limits",
        "",
        "The target is whiff among recorded swing outcomes; takes are excluded. These are predictive comparisons, not causal estimates.",
        "The release-speed and observed tracking blocks use measurements from the current pitch. They are diagnostic comparisons and are not eligible for a pre-pitch recommendation unless replaced with historical, pre-pitch estimates.",
        "Pitcher and batter IDs may help on recurring players but can memorize the 2024 population. This evaluation does not test generalization to entirely new players.",
        "Choose feature blocks based on performance and stability across all rolling windows, plus baseball availability and interpretability. Do not use the reserved final holdout repeatedly for selection.",
        "",
        "Fold-level results and exact feature lists are in `feature_block_scores.csv`; aggregated results are in `feature_block_summary.csv`.",
    ])
    report = "\n".join(lines) + "\n"
    (output_dir / "feature_selection_summary.md").write_text(report, encoding="utf-8")
    return fold_scores, summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/statcast.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    _, _, report = compare(args.input, args.output)
    print(report)


if __name__ == "__main__":
    main()
