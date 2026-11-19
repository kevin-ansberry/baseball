[README.md](https://github.com/user-attachments/files/35100129/README.md)
# Pitch Mix Lab

**A public-data baseball research project on pitch selection and decision support.**

Pitch Mix Lab asks whether pitch outcomes vary with pitch type, count, batter handedness, and pitcher characteristics. It is designed as a reproducible portfolio project: acquire public Statcast data, validate it, build a time-aware baseline, and communicate the limits of the result.

This is an independent analysis using public data. It does not represent the Cleveland Guardians or use any club's proprietary information.

## Research question

> How do pitch outcomes differ by pitch type, count, batter handedness, and pitcher, and how well can those outcomes be predicted on a later time period?

The first version focuses on **whiff probability** as a measurable outcome. It is a useful starting point, not a complete measure of pitch value: called strikes, contact quality, walks, sequencing, catcher targets, and strategic selection also matter. Later iterations can add run value and model uncertainty.

## Why this project

The work demonstrates a complete baseball analytics workflow: data ingestion, quality checks, feature engineering, a baseline model, temporal evaluation, and a concise interpretation for a baseball audience. Recommendations will be framed as descriptive associations unless a later design supports causal claims.

## Quick start

Requires Python 3.10+.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pitch_mix_lab.fetch --start 2024-04-01 --end 2024-09-30 --output data/raw/statcast.csv
.\.venv\Scripts\python.exe -m pitch_mix_lab.analyze --input data/raw/statcast.csv --output reports --min-pitches 50
.\.venv\Scripts\python.exe -m pitch_mix_lab.compare_features --input data/raw/statcast.csv --output reports
.\.venv\Scripts\python.exe -m pitch_mix_lab.compare_models --input data/raw/statcast.csv --output reports
```

The data fetch uses the public Baseball Savant Statcast feed through `pybaseball`. Fetch a short date range first; a full season can take time and the source may throttle requests. Raw data is intentionally not committed to Git.

## Method

1. Fetch pitch-level Statcast records for a specified date range.
2. Validate required columns and filter to pitches with a usable pitch type and outcome.
3. Define a whiff as a swinging strike or missed bunt (`swinging_strike`, `swinging_strike_blocked`, or `missed_bunt`). Foul and in-play swings are non-whiffs; takes are excluded.
4. Train a simple logistic-regression baseline using pitch type, count, handedness, and release speed when available.
5. Evaluate on the latest 20% of dates, keeping the split chronological. Report ROC AUC, Brier score, and a calibration plot so discrimination and probability quality are both visible.
6. Compare descriptive whiff rates by pitch type, count, batter handedness, and pitcher handedness. The rate table includes 95% Wilson intervals and a configurable minimum-sample flag. Do not interpret these contrasts as causal effects of changing a pitcher's mix.
7. Compare candidate feature blocks on three expanding-window validation periods before the reserved final holdout. Review mean and fold-level ROC AUC and Brier scores in the feature comparison reports. Current-pitch tracking features are labeled as diagnostic because they are observed during or after delivery.
8. Compare L2/L1 logistic regression and histogram gradient boosting on the same pre-pitch feature sets and rolling date windows. Keep the latest-date holdout reserved until model choice is complete.

## Repository layout

```text
src/pitch_mix_lab/  data fetch, validation, analysis, and feature comparison code
data/raw/           downloaded data (gitignored)
reports/            generated figures, summaries, rate tables, and feature/model comparisons
docs/               research notes and limitations
```

## Current limitations

- Whiff probability alone is not run prevention or pitch value.
- Pitch selection is not random. Differences between pitch types can reflect pitcher skill, count, batter, location, and strategic intent.
- Public data and a simple baseline cannot reproduce the information available to a professional club.
- A model score is not evidence that a recommended usage change would improve outcomes.
- Confidence intervals describe binomial sampling uncertainty; they do not account for pitcher, batter, or pitch-selection confounding.
- Current-pitch speed, location, and movement measurements are not known before the pitch; they must be replaced with historical estimates for a pre-pitch recommendation.
- Pitcher and batter ID features can memorize recurring players and do not establish performance on entirely new players.
- The model comparison currently uses fixed starter settings. It is a comparison, not a complete hyperparameter search.

## Next steps

- Add pitcher-level and batter-level partial pooling, with uncertainty intervals.
- Build historical pitcher/pitch-type arsenal features using only prior pitches, then compare them on rolling time windows.
- Tune selected model families within the development period and evaluate the final selected pipeline once on the untouched holdout.
- Explore run value with an explicit, documented outcome definition.
- Build a small interactive view for one pitcher and expose sample sizes and uncertainty alongside every rate.
- Write a one-page baseball-facing memo with a concrete example and caveats.

## Data attribution

Statcast data is served publicly by [Baseball Savant](https://baseballsavant.mlb.com/). `pybaseball` provides a convenient interface. Respect source availability and rate limits; see [pybaseball documentation](https://github.com/jldbc/pybaseball).

## Model comparison

From the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m pitch_mix_lab.compare_models --input data/raw/statcast.csv --output reports
