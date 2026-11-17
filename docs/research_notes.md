# Research notes

## Initial hypothesis

The probability of a swinging strike varies across pitch type and count. Any useful comparison must account for handedness and avoid presenting raw pitch-type averages as if the pitches had been assigned at random.

## Outcome definition

The first analysis includes pitches with a recorded swing outcome. A whiff is `swinging_strike` or `swinging_strike_blocked`; foul and in-play outcomes are non-whiffs. Takes are excluded from the denominator. Report the resulting sample size because the definition changes what the reported rate means.

## Evaluation design

Use a chronological holdout so the test set follows the training data in time. Compare a logistic-regression baseline using pitch type, count, batter handedness, pitcher throwing hand, and release speed. Report ROC AUC and Brier score, and inspect the calibration plot. This evaluates prediction, not whether following a pitch-mix suggestion causes better results.

## Questions for iteration

- Is the test period long enough to include meaningful variation in pitchers and hitters?
- Should pitcher identity be added as a fixed effect or modeled with partial pooling?
- Can location and movement improve calibration without obscuring interpretation?
- How should pitch value account for called strikes and contact quality as well as whiffs?
