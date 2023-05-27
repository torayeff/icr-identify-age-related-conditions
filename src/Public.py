import optuna.integration.lightgbm as lgb
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
test_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")
greeks_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/greeks.csv")
sample_submission_df = pd.read_csv(
    "/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv"
)

numerical_features = [
    "AB",
    "AF",
    "AH",
    "AM",
    "AR",
    "AX",
    "AY",
    "AZ",
    "BC",
    "BD",
    "BN",
    "BP",
    "BQ",
    "BR",
    "BZ",
    "CB",
    "CC",
    "CD",
    "CF",
    "CH",
    "CL",
    "CR",
    "CS",
    "CU",
    "CW",
    "DA",
    "DE",
    "DF",
    "DH",
    "DI",
    "DL",
    "DN",
    "DU",
    "DV",
    "DY",
    "EB",
    "EE",
    "EG",
    "EH",
    "EL",
    "EP",
    "EU",
    "FC",
    "FD",
    "FE",
    "FI",
    "FL",
    "FR",
    "FS",
    "GB",
    "GE",
    "GF",
    "GH",
    "GI",
    "GL",
]
categorical_features = ["EJ"]
features = numerical_features + categorical_features

train_df["EJ"] = train_df["EJ"].replace({"A": 0, "B": 1})
test_df["EJ"] = test_df["EJ"].replace({"A": 0, "B": 1})

X = train_df.drop("Class", axis=1)
y = train_df["Class"]

X_test = test_df.copy()

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
}

for i in range(50):

    positive_count_train = y.value_counts()[1]
    sampler = RandomUnderSampler(
        sampling_strategy={0: positive_count_train, 1: positive_count_train},
        random_state=i,
        replacement=True,
    )
    X_re, y_re = sampler.fit_resample(X, y)
    (X_train, X_val, y_train, y_val) = train_test_split(
        X_re, y_re, test_size=0.2, random_state=42
    )
    lgb_eval = lgb.Dataset(X_val.drop("Id", axis=1), y_val, free_raw_data=False)
    lgb_train = lgb.Dataset(X_re.drop("Id", axis=1), y_re, free_raw_data=False)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        categorical_feature=categorical_features,
        num_boost_round=1000,
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    pred = model.predict(X_test.drop("Id", axis=1), num_iteration=model.best_iteration)
    if i == 0:
        output = pd.DataFrame(pred, columns=["pred" + str(i + 1)])
        output2 = output
    else:
        output = pd.DataFrame(pred, columns=["pred" + str(i + 1)])
        output2 = pd.concat([output2, output], axis=1)

pred = output2.mean(axis="columns")
submit = pd.DataFrame(test_df["Id"], columns=["Id"])
submit["class_0"] = 1 - pred
submit["class_1"] = pred
submit.to_csv("submission.csv", index=False)
