from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

data_path = Path("../data/")
seed = 42


def balanced_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    n0, n1 = np.bincount(y_true.astype(int))
    w0 = 1 / (n0 / len(y_true))
    w1 = 1 / (n1 / len(y_true))

    l0 = -w0 / n0 * np.sum(np.where(y_true == 0, 1, 0) * np.log(1 - y_pred))
    l1 = -w1 / n1 * np.sum(np.where(y_true != 0, 1, 0) * np.log(y_pred))

    return (l0 + l1) / (w0 + w1)


def lgb_metric(y_true, y_pred):
    return "balanced_log_loss", balanced_log_loss(y_true, y_pred), False


# read data
train_df = pd.read_csv(data_path / "train.csv")
test_df = pd.read_csv(data_path / "test.csv")
test_ids = test_df["Id"]

# preprocess data
feature_columns = train_df.columns[1:-1]

# Convert the categorical feature to numeric representation using label encoding
label_encoder = LabelEncoder()
train_df["EJ"] = label_encoder.fit_transform(train_df["EJ"])
test_df["EJ"] = label_encoder.transform(test_df["EJ"])

X_train, y_train = train_df[feature_columns], train_df["Class"]
X_test = test_df[feature_columns]

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
test_probs = np.zeros((len(test_df), 2))

cur_split = 1
for train_idx, val_idx in kf.split(X_train, y_train):
    print(f"Fold: {cur_split}".center(100, "-"))
    cur_split += 1

    X_train_fold, y_train_fold = (X_train.iloc[train_idx], y_train.iloc[train_idx])
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    params = {
        "objective": "binary",
        "n_estimators": 10000,
        "n_jobs": -1,
        "verbose": -1,
        "seed": seed,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric=["binary_logloss", lgb_metric],
        callbacks=[
            lgb.log_evaluation(period=-1),
            lgb.early_stopping(stopping_rounds=5, verbose=True),
        ],
    )

    test_fold_probs = model.predict_proba(X_test)
    test_probs += test_fold_probs / n_splits

sub_df = pd.DataFrame(
    {"Id": test_ids, "Class_0": test_probs[:, 0], "Class_1": test_probs[:, 1]}
)
sub_df.to_csv("submission.csv", index=False)
