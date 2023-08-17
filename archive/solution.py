from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

data_path = Path("/kaggle/input/icr-identify-age-related-conditions")
seed = 42


def balanced_log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    n0, n1 = np.bincount(y_true.astype(int))
    w0 = 1 / (n0 / len(y_true))
    w1 = 1 / (n1 / len(y_true))

    l0 = -w0 / n0 * np.sum(np.where(y_true == 0, 1, 0) * np.log(1 - y_pred))
    l1 = -w1 / n1 * np.sum(np.where(y_true != 0, 1, 0) * np.log(y_pred))

    return (l0 + l1) / (w0 + w1)


# read data
train_df = pd.read_csv(data_path / "train.csv")
test_df = pd.read_csv(data_path / "test.csv")
greeks_df = pd.read_csv(data_path / "greeks.csv")

# prepare data
feature_cols = train_df.columns.tolist()[1:-1]
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["Class"], random_state=42
)
greeks_df = greeks_df.loc[train_df.index]
train_df = train_df.reset_index(drop=True)
greeks_df = greeks_df.reset_index(drop=True)
train_df.drop(columns=["Id"], inplace=True)

# undersample
under_sampler = RandomUnderSampler(random_state=seed)
train_df, _ = under_sampler.fit_resample(train_df, train_df["Class"])

# train
oof = np.zeros(len(train_df))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
test_probs = []
val_preds = []
fold = 1
for train_idx, val_idx in skf.split(train_df, train_df["Class"]):
    print(f"Fold-{fold}".center(110, "-"))
    fold += 1

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
        "early_stopping_rounds": 1000,
        "use_best_model": True,
        "random_seed": seed,
    }

    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], cat_features=["EJ"], verbose=1000
    )

    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    val_preds.append(model.predict_proba(val_df[feature_cols])[:, 1])
    test_preds = model.predict_proba(test_df.iloc[:, 1:])
    test_probs.append(test_preds)

print(f"OOF score: {balanced_log_loss(train_df['Class'], oof):.4f}")
print(f"CV score: {balanced_log_loss(val_df['Class'], np.mean(val_preds, axis=0)):.4f}")

# generate a submission file
test_probs = np.mean(test_probs, axis=0)
sub_df = pd.DataFrame(
    {"Id": test_df.Id, "Class_0": test_probs[:, 0], "Class_1": test_probs[:, 1]}
)
sub_df.to_csv("submission.csv", index=False)
