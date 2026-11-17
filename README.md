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
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pitch_mix_lab.fetch --start 2024-04-01 --end 2024-09-30 --output data/raw/statcast.csv
python -m pitch_mix_lab.analyze --input data/raw/statcast.csv --output reports
```

The data fetch uses the public Baseball Savant Statcast feed through `pybaseball`. Fetch a short date range first; a full season can take time and the source may throttle requests. Raw data is intentionally not committed to Git.

## Method

1. Fetch pitch-level Statcast records for a specified date range.
2. Validate required columns and filter to pitches with a usable pitch type and outcome.
3. Define a whiff as a swinging strike or missed bunt (`swinging_strike`, `swinging_strike_blocked`, or `missed_bunt`). Foul and in-play swings are non-whiffs; takes are excluded.
4. Train a simple logistic-regression baseline using pitch type, count, handedness, and release speed when available.
5. Evaluate on the latest 20% of dates, keeping the split chronological. Report ROC AUC, Brier score, and a calibration plot so discrimination and probability quality are both visible.
6. Compare descriptive whiff rates by pitch type and count. Do not interpret these contrasts as causal effects of changing a pitcher's mix.

## Repository layout

```text
src/pitch_mix_lab/  data fetch, validation, and analysis code
data/raw/           downloaded data (gitignored)
reports/            generated figures and summary
docs/               research notes and limitations
```

## Current limitations

- Whiff probability alone is not run prevention or pitch value.
- Pitch selection is not random. Differences between pitch types can reflect pitcher skill, count, batter, location, and strategic intent.
- Public data and a simple baseline cannot reproduce the information available to a professional club.
- A model score is not evidence that a recommended usage change would improve outcomes.

## Next steps

- Add pitcher-level and batter-level partial pooling, with uncertainty intervals.
- Add location and movement features, then test whether they improve out-of-time calibration.
- Explore run value with an explicit, documented outcome definition.
- Build a small interactive view for one pitcher and expose sample sizes and uncertainty alongside every rate.
- Write a one-page baseball-facing memo with a concrete example and caveats.

## Data attribution

Statcast data is served publicly by [Baseball Savant](https://baseballsavant.mlb.com/). `pybaseball` provides a convenient interface. Respect source availability and rate limits; see [pybaseball documentation](https://github.com/jldbc/pybaseball).
