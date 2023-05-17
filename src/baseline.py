from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
greeks_df = pd.read_csv(data_path / "greeks.csv")

# some columns have trailing spaces
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()
feature_cols = train_df.columns.tolist()[1:-1]

# Encode categorical column
label_encoder = LabelEncoder()
train_df["EJ"] = label_encoder.fit_transform(train_df["EJ"])
test_df["EJ"] = label_encoder.transform(test_df["EJ"])

# training
oof = np.zeros(len(train_df))
skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
test_probs = []

for train_idx, val_idx in skf.split(train_df, greeks_df.iloc[:, 1:-1]):

    X_train, y_train = (
        train_df.loc[train_idx, feature_cols],
        train_df.loc[train_idx, "Class"],
    )

    X_val, y_val = (
        train_df.loc[val_idx, feature_cols],
        train_df.loc[val_idx, "Class"],
    )

    params = {
        "iterations": 10000,
        "learning_rate": 0.005,
        "early_stopping_rounds": 1000,
        "auto_class_weights": "Balanced",
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass:use_weights=False",
        "random_seed": 42,
        "use_best_model": True,
        "l2_leaf_reg": 1,
        "max_ctr_complexity": 15,
        "max_depth": 10,
        "grow_policy": "Lossguide",
        "max_leaves": 64,
        "min_data_in_leaf": 40,
    }
    model = cb.CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)
    preds = model.predict_proba(X_val)
    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    test_probs.append(model.predict_proba(test_df.iloc[:, 1:]))

print(f"OOF score: {balanced_log_loss(train_df['Class'], oof):.4f}")

# generate a submission file
test_probs = np.mean(test_probs, axis=0)
sub_df = pd.DataFrame(
    {"Id": test_df.Id, "Class_0": test_probs[:, 0], "Class_1": test_probs[:, 1]}
)
sub_df.to_csv("submission.csv", index=False)
