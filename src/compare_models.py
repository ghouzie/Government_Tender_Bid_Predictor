from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR

from helpers import regression_metrics
from model_tools import (
    add_log_columns,
    make_lightgbm,
    make_preprocessor,
    to_bhd,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/tier2_with_history.csv"
RESULT_FILE = ROOT / "results/model_comparison.csv"


def main():
    data = pd.read_csv(DATA_FILE)
    data = add_log_columns(data)

    train = data[
        data["year"] <= 2023
    ].copy()

    validation = data[
        data["year"] == 2024
    ].copy()

    preprocessor = make_preprocessor()

    x_train = preprocessor.fit_transform(train)
    x_validation = preprocessor.transform(validation)

    y_train = np.log1p(
        train["winning_bid_bhd"]
    )

    models = {
        "Ridge": Ridge(
            alpha=3,
            solver="lsqr",
        ),
        "Linear SVR": LinearSVR(
            C=1.0,
            epsilon=0.1,
            max_iter=5000,
            random_state=42,
        ),
        "LightGBM": make_lightgbm(),
    }

    predictions = {}
    rows = []

    for name, model in models.items():
        print("Training", name)

        model.fit(
            x_train,
            y_train,
        )

        predicted = to_bhd(
            model.predict(x_validation)
        )

        predictions[name] = predicted

        score = regression_metrics(
            validation["winning_bid_bhd"],
            predicted,
        )

        rows.append({
            "model": name,
            **score,
        })

    # We also test a simple mixture of the linear and tree models.
    ridge = predictions["Ridge"]
    lightgbm = predictions["LightGBM"]

    blend = np.expm1(
        0.2 * np.log1p(ridge)
        + 0.8 * np.log1p(lightgbm)
    )

    score = regression_metrics(
        validation["winning_bid_bhd"],
        blend,
    )

    rows.append({
        "model": "Ridge + LightGBM",
        **score,
    })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "median_percentage_error"
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
    )

    print()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
