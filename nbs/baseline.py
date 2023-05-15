from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

data_path = Path("../data/")


def balanced_log_loss(y_true, pred_probs):
    pred_probs = np.maximum(np.minimum(pred_probs, 1 - 1e-15), 1e-15)
    pred_probs = pred_probs / np.sum(pred_probs, axis=1)[:, None]

    n0, n1 = np.bincount(y_true)
    w0 = 1 / (n0 / len(y_true))
    w1 = 1 / (n1 / len(y_true))

    l0 = -w0 / n0 * np.sum(np.where(y_true == 0, 1, 0) * np.log(pred_probs[:, 0]))
    l1 = -w1 / n1 * np.sum(np.where(y_true == 1, 1, 0) * np.log(pred_probs[:, 1]))

    return (l0 + l1) / (w0 + w1)


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
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_probs = np.zeros((len(test_df), 2))

for train_idx, val_idx in kf.split(X_train, y_train):
    X_train_fold, y_train_fold = (X_train.iloc[train_idx], y_train.iloc[train_idx])
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(n_estimators=1000, **params)
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=50,
        verbose=False,
    )

    val_probs = model.predict_proba(X_val_fold)
    val_pred = model.predict(X_val_fold)

    acc = accuracy_score(y_val_fold, val_pred)
    print(
        f"Validation accuracy: {acc:.4f}.",
        f"Balanced log loss: {balanced_log_loss(y_val_fold, val_probs):.4f}",
    )

    test_fold_probs = model.predict_proba(X_test)
    test_probs += test_fold_probs / n_splits

sub_df = pd.DataFrame(
    {"Id": test_ids, "Class_0": test_probs[:, 0], "Class_1": test_probs[:, 1]}
)
sub_df.to_csv("submission.csv", index=False)
