from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from helpers import regression_metrics
from model_tools import (
    add_log_columns,
    make_lightgbm,
    make_preprocessor,
    to_bhd,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data/processed/tier2_with_history.csv"
HISTORY_FILE = ROOT / "data/processed/historical_awards.csv"

MODEL_FILE = ROOT / "models/final_model.joblib"
HISTORY_MODEL_FILE = ROOT / "models/history_model.joblib"

RESULT_FILE = ROOT / "results/final_metrics.json"
PREDICTION_FILE = ROOT / "results/test_predictions_2025.csv"


def train_history_model(history):
    history = history[
        history["year"] <= 2024
    ].copy()

    preprocessor = ColumnTransformer([
        (
            "categories",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
            ),
            ["entity", "sector"],
        ),
        (
            "subject",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=10000,
                sublinear_tf=True,
            ),
            "subject",
        ),
    ])

    x = preprocessor.fit_transform(history)

    model = Ridge(
        alpha=3,
        solver="lsqr",
    )

    model.fit(
        x,
        np.log1p(
            history["winning_bid_bhd"]
        ),
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
        },
        HISTORY_MODEL_FILE,
    )


def main():
    data = pd.read_csv(DATA_FILE)
    history = pd.read_csv(HISTORY_FILE)

    data = add_log_columns(data)

    train = data[
        data["year"] <= 2023
    ].copy()

    validation = data[
        data["year"] == 2024
    ].copy()

    test = data[
        data["year"] == 2025
    ].copy()

    # Step 1:
    # Train on 2021-2023 and use 2024 to measure
    # errors and choose the prediction range.
    preprocessor = make_preprocessor()

    x_train = preprocessor.fit_transform(train)
    x_validation = preprocessor.transform(validation)

    model = make_lightgbm()

    model.fit(
        x_train,
        np.log1p(
            train["winning_bid_bhd"]
        ),
    )

    validation_prediction = to_bhd(
        model.predict(x_validation)
    )

    validation_error = np.abs(
        np.log(
            np.maximum(
                validation_prediction,
                0.01,
            )
        )
        - np.log(
            validation["winning_bid_bhd"]
        )
    )

    range_factor = float(
        np.exp(
            np.quantile(
                validation_error,
                0.80,
            )
        )
    )

    # Step 2:
    # Once the model choice is fixed, train again
    # using everything through 2024.
    train_final = data[
        data["year"] <= 2024
    ].copy()

    preprocessor_final = make_preprocessor()

    x_final = preprocessor_final.fit_transform(
        train_final
    )

    x_test = preprocessor_final.transform(
        test
    )

    model_final = make_lightgbm()

    model_final.fit(
        x_final,
        np.log1p(
            train_final["winning_bid_bhd"]
        ),
    )

    test_prediction = to_bhd(
        model_final.predict(x_test)
    )

    score = regression_metrics(
        test["winning_bid_bhd"],
        test_prediction,
    )

    lower = test_prediction / range_factor
    upper = test_prediction * range_factor

    range_coverage = float(
        np.mean(
            (
                test["winning_bid_bhd"]
                >= lower
            )
            & (
                test["winning_bid_bhd"]
                <= upper
            )
        )
    )

    joblib.dump(
        {
            "preprocessor": preprocessor_final,
            "model": model_final,
            "range_factor": range_factor,
        },
        MODEL_FILE,
    )

    train_history_model(history)

    predictions = test[[
        "reference",
        "year",
        "entity",
        "subject",
        "winning_bid_bhd",
    ]].copy()

    predictions["prediction_bhd"] = test_prediction
    predictions["range_lower"] = lower
    predictions["range_upper"] = upper

    predictions["percentage_error"] = (
        np.abs(
            predictions["prediction_bhd"]
            - predictions["winning_bid_bhd"]
        )
        / predictions["winning_bid_bhd"]
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    output = {
        "training_rows_2021_2023": len(train),
        "validation_rows_2024": len(validation),
        "test_rows_2025": len(test),
        "production_training_rows_2021_2024": len(train_final),
        "range_factor_80_percent": range_factor,
        "range_coverage_2025": range_coverage,
        **score,
    }

    RESULT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
