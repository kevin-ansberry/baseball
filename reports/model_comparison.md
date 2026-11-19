# Model comparison

The final chronological holdout begins 2024-08-25 and was not used for model or feature selection.
Candidates were compared on 3 expanding-window periods drawn from earlier dates, using two feature sets and the same splits.
The target is whiff among recorded swing outcomes; takes are excluded.

## Summary

| Feature set | Model | ROC AUC mean (SD) | Brier mean (SD) | Calibration gap | Brier delta vs current logistic |
|---|---|---:|---:|---:|---:|
| pre_pitch_baseline | hist_gradient_boosting | 0.628 (0.003) | 0.1712 (0.0010) | 0.0065 | -0.0003 |
| pre_pitch_baseline | logistic_l1_c01 | 0.623 (0.003) | 0.1716 (0.0010) | 0.0073 | +0.0000 |
| pre_pitch_baseline | logistic_l2_c01 | 0.623 (0.003) | 0.1715 (0.0010) | 0.0070 | +0.0000 |
| pre_pitch_baseline | logistic_l2_c1 | 0.623 (0.003) | 0.1715 (0.0010) | 0.0073 | +0.0000 |
| pre_pitch_plus_game_state | hist_gradient_boosting | 0.631 (0.002) | 0.1709 (0.0011) | 0.0070 | -0.0006 |
| pre_pitch_plus_game_state | logistic_l1_c01 | 0.626 (0.002) | 0.1713 (0.0011) | 0.0055 | -0.0002 |
| pre_pitch_plus_game_state | logistic_l2_c01 | 0.626 (0.002) | 0.1713 (0.0011) | 0.0059 | -0.0002 |
| pre_pitch_plus_game_state | logistic_l2_c1 | 0.626 (0.002) | 0.1713 (0.0011) | 0.0060 | -0.0002 |

Higher ROC AUC is better; lower Brier score and calibration gap are better. Negative Brier delta means lower average Brier error than the current L2 logistic model with the pre-pitch baseline features.
The two feature sets are pre-pitch baseline and pre-pitch baseline plus game state. Model comparisons are fair within each feature set; compare across feature sets separately.

## Scope and limits

The logistic models use one-hot encoded features. L1/L2 settings are fixed starter comparisons, not a hyperparameter search. The histogram gradient-boosting model uses dense one-hot inputs and fixed starter settings.
The rolling-window scores are model-selection evidence, not final performance. After choosing a model and feature set, evaluate it once on the reserved final holdout.
This experiment does not establish causal pitch value or performance on entirely new pitchers and batters. It evaluates whiffs conditional on a recorded swing outcome.

Fold-level results are in `model_comparison_folds.csv`; aggregate results are in `model_comparison_summary.csv`.
