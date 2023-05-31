# Observations

## 2023-05-31
The CV is very sensitive to the random seed parameter.

## 2023-05-28
1. PCA did not help
2. Manual imputing also did not help
---

Inconsistent public leaderboard and local OOF score:

OOF: 0.3971 | PL: 0.22
OOF: 0.3462 | PL: 0.23

Similar problem is happening here https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/412638

Undersampler seed 42
OOF score: 0.3125
CV score: 0.2881
LB score: 0.22

Undersampler seed 19
OOF score: 0.2565
CV score: 0.2479
LB score: 0.20

NB1: Changing seed to 19 gives a good improvement, but I am highly suspicious...

NB2: Can I rely on CV score?

25 x 5 (kfold) ensembler + undersample