"""Compare candidate classifiers on shared, expanding-window date splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pitch_mix_lab.analyze import build_model, prepare
from pitch_mix_lab.compare_features import (
    BASE_CATEGORICAL,
    GAME_STATE_CATEGORICAL,
    GAME_STATE_NUMERIC,
    rolling_date_splits,
)


def feature_sets() -> list[dict[str, object]]:
    """Model comparison uses identical feature sets for each candidate classifier."""
    return [
        {
            "feature_set": "pre_pitch_baseline",
            "availability": "Candidate pre-pitch features",
            "categorical": BASE_CATEGORICAL,
            "numeric": [],
        },
        {
            "feature_set": "pre_pitch_plus_game_state",
            "availability": "Candidate pre-pitch features",
            "categorical": BASE_CATEGORICAL + GAME_STATE_CATEGORICAL,
            "numeric": GAME_STATE_NUMERIC,
        },
    ]


def model_specs() -> list[dict[str, object]]:
    """Return baseline and regularized linear models plus a nonlinear tree model."""
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    modern_logistic = version >= (1, 8)
    l2_params = (
        {"model__C": 1.0, "model__l1_ratio": 0.0, "model__solver": "lbfgs"}
        if modern_logistic else
        {"model__C": 1.0, "model__penalty": "l2", "model__solver": "lbfgs"}
    )
    l2_strong_params = {**l2_params, "model__C": 0.1}
    l1_params = (
        {"model__C": 0.1, "model__l1_ratio": 1.0, "model__solver": "saga", "model__max_iter": 3000}
        if modern_logistic else
        {"model__C": 0.1, "model__penalty": "l1", "model__solver": "liblinear"}
    )
    return [
        {
            "model": "logistic_l2_c1",
            "family": "Logistic regression",
            "description": "Current L2 logistic baseline (C=1)",
            "params": l2_params,
            "kind": "linear",
        },
        {
            "model": "logistic_l2_c01",
            "family": "Logistic regression",
            "description": "Stronger L2 shrinkage (C=0.1)",
            "params": l2_strong_params,
            "kind": "linear",
        },
        {
            "model": "logistic_l1_c01",
            "family": "Logistic regression",
            "description": "Sparse L1 logistic (C=0.1)",
            "params": l1_params,
            "kind": "linear",
        },
        {
            "model": "hist_gradient_boosting",
            "family": "Histogram gradient boosting",
            "description": "Nonlinear boosted trees (120 iterations, 15 leaves)",
            "params": {},
            "kind": "hist_gradient_boosting",
        },
    ]


def make_model(spec: dict[str, object], categorical: list[str], numeric: list[str]) -> Pipeline:
    if spec["kind"] == "linear":
        model = build_model(categorical=categorical, numeric=numeric)
        model.set_params(**spec["params"])
        return model

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [("categorical", categorical_pipeline, categorical)]
    if numeric:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric))
    features = ColumnTransformer(transformers, sparse_threshold=0.0)
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=17,
    )
    return Pipeline([("features", features), ("model", model)])


def calibration_gap(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> float:
    """Mean absolute calibration gap across quantile bins."""
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(observed - predicted)))


def compare(input_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw = pd.read_csv(input_path, low_memory=False)
    frame = prepare(raw)
    if frame["whiff"].nunique() != 2 or len(frame) < 100:
        raise ValueError("Need at least 100 swing outcomes and both whiffs and contact to compare models.")

    # Preserve the final chronological holdout for a single evaluation after model choice.
    final_holdout_start = frame["game_date"].quantile(0.8)
    development = frame[frame["game_date"] < final_holdout_start].copy()
    splits = rolling_date_splits(development)
    candidates = feature_sets()
    models = model_specs()

    candidate_columns = set()
    for feature_set in candidates:
        candidate_columns.update(feature_set["categorical"])
        candidate_columns.update(feature_set["numeric"])
    missing = candidate_columns - set(development.columns)
    if missing:
        raise ValueError(f"Input is missing candidate features: {', '.join(sorted(missing))}")

    records: list[dict[str, object]] = []
    for fold, (valid_start, valid_end) in enumerate(splits, start=1):
        train = development[development["game_date"] < valid_start]
        valid = development[(development["game_date"] >= valid_start) &
                            (development["game_date"] < valid_end)]
        constant_rate = float(train["whiff"].mean())
        constant_brier = brier_score_loss(valid["whiff"], np.full(len(valid), constant_rate))

        for feature_set in candidates:
            categorical = list(feature_set["categorical"])
            numeric = list(feature_set["numeric"])
            predictors = categorical + numeric
            for spec in models:
                estimator = make_model(spec, categorical, numeric)
                estimator.fit(train[predictors], train["whiff"])
                probabilities = estimator.predict_proba(valid[predictors])[:, 1]
                records.append({
                    "fold": fold,
                    "train_start": train["game_date"].min().date().isoformat(),
                    "train_end": train["game_date"].max().date().isoformat(),
                    "validation_start": valid["game_date"].min().date().isoformat(),
                    "validation_end": valid["game_date"].max().date().isoformat(),
                    "train_rows": len(train),
                    "validation_rows": len(valid),
                    "feature_set": feature_set["feature_set"],
                    "availability": feature_set["availability"],
                    "model": spec["model"],
                    "model_family": spec["family"],
                    "model_description": spec["description"],
                    "categorical_features": ";".join(categorical),
                    "numeric_features": ";".join(numeric),
                    "roc_auc": roc_auc_score(valid["whiff"], probabilities),
                    "brier_score": brier_score_loss(valid["whiff"], probabilities),
                    "constant_rate_brier": constant_brier,
                    "calibration_gap": calibration_gap(valid["whiff"], probabilities),
                })

    fold_scores = pd.DataFrame(records)
    current_key = (fold_scores["feature_set"] == "pre_pitch_baseline") & \
                  (fold_scores["model"] == "logistic_l2_c1")
    current_brier = fold_scores[current_key].set_index("fold")["brier_score"]
    fold_scores["brier_delta_vs_current_logistic"] = fold_scores.apply(
        lambda row: row["brier_score"] - current_brier.loc[row["fold"]], axis=1
    )
    summary = (fold_scores.groupby(
        ["feature_set", "availability", "model", "model_family", "model_description"], as_index=False
    ).agg(folds=("fold", "nunique"), mean_roc_auc=("roc_auc", "mean"), sd_roc_auc=("roc_auc", "std"),
          mean_brier=("brier_score", "mean"), sd_brier=("brier_score", "std"),
          mean_constant_brier=("constant_rate_brier", "mean"), mean_calibration_gap=("calibration_gap", "mean"),
          mean_brier_delta_vs_current_logistic=("brier_delta_vs_current_logistic", "mean")))

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output_dir / "model_comparison_folds.csv", index=False)
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    lines = [
        "# Model comparison",
        "",
        f"The final chronological holdout begins {final_holdout_start.date()} and was not used for model or feature selection.",
        f"Candidates were compared on {len(splits)} expanding-window periods drawn from earlier dates, using two feature sets and the same splits.",
        "The target is whiff among recorded swing outcomes; takes are excluded.",
        "",
        "## Summary",
        "",
        "| Feature set | Model | ROC AUC mean (SD) | Brier mean (SD) | Calibration gap | Brier delta vs current logistic |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        auc_sd = row.sd_roc_auc if pd.notna(row.sd_roc_auc) else 0.0
        brier_sd = row.sd_brier if pd.notna(row.sd_brier) else 0.0
        lines.append(
            f"| {row.feature_set} | {row.model} | {row.mean_roc_auc:.3f} ({auc_sd:.3f}) | "
            f"{row.mean_brier:.4f} ({brier_sd:.4f}) | {row.mean_calibration_gap:.4f} | "
            f"{row.mean_brier_delta_vs_current_logistic:+.4f} |"
        )
    lines.extend([
        "",
        "Higher ROC AUC is better; lower Brier score and calibration gap are better. Negative Brier delta means lower average Brier error than the current L2 logistic model with the pre-pitch baseline features.",
        "The two feature sets are pre-pitch baseline and pre-pitch baseline plus game state. Model comparisons are fair within each feature set; compare across feature sets separately.",
        "",
        "## Scope and limits",
        "",
        "The logistic models use one-hot encoded features. L1/L2 settings are fixed starter comparisons, not a hyperparameter search. The histogram gradient-boosting model uses dense one-hot inputs and fixed starter settings.",
        "The rolling-window scores are model-selection evidence, not final performance. After choosing a model and feature set, evaluate it once on the reserved final holdout.",
        "This experiment does not establish causal pitch value or performance on entirely new pitchers and batters. It evaluates whiffs conditional on a recorded swing outcome.",
        "",
        "Fold-level results are in `model_comparison_folds.csv`; aggregate results are in `model_comparison_summary.csv`.",
    ])
    report = "\n".join(lines) + "\n"
    (output_dir / "model_comparison.md").write_text(report, encoding="utf-8")
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
