"""Create deterministic, preprocessed splits for version comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--preprocess-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--problem", required=True, choices=("classification", "regression"))
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()

    with args.preprocess_config.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)

    data = pd.read_csv(args.data)
    categorical = config["categorical_cols"]
    numerical = config["numerical_cols"]
    encoding = config["encoding"]
    features = data[numerical + categorical]
    target = data.iloc[:, -1]

    if encoding.get("target") == "label":
        target = pd.Series(LabelEncoder().fit_transform(target), index=data.index)

    stratify = target if args.problem == "classification" else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=args.seed,
        stratify=stratify,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_temp if args.problem == "classification" else None,
    )

    transformers = []
    numerical_encoding = encoding.get("numerical_features")
    if numerical_encoding == "minmax":
        transformers.append(("num", MinMaxScaler(), numerical))
    elif numerical_encoding == "standard":
        transformers.append(("num", StandardScaler(), numerical))

    if categorical and encoding.get("categorical_features") == "onehot":
        transformers.append(
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical)
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    train_values = preprocessor.fit_transform(x_train)
    val_values = preprocessor.transform(x_val)
    test_values = preprocessor.transform(x_test)
    columns = list(preprocessor.get_feature_names_out())

    args.output.mkdir(parents=True, exist_ok=True)
    for name, values, labels, index in (
        ("train", train_values, y_train, x_train.index),
        ("val", val_values, y_val, x_val.index),
        ("test", test_values, y_test, x_test.index),
    ):
        frame = pd.DataFrame(values, columns=columns, index=index)
        frame[data.columns[-1]] = pd.Series(labels, index=index)
        frame.to_csv(args.output / f"{name}.csv", index=False)
        print(f"{name}: {frame.shape}")


if __name__ == "__main__":
    main()
