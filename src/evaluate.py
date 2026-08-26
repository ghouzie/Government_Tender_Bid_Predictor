import json

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RIDGE_ALPHAS, TIER1_DATA, TIER2_DATA
from data import load_modeling_data, prepare_tier2_features
from model import build_tier1_model, build_tier2_model, predict_bhd


def split_by_time(df):
    train = df[df["period_dt"] < "2024-01-01"].copy()
    validation = df[
        (df["period_dt"] >= "2024-01-01")
        & (df["period_dt"] < "2025-01-01")
    ].copy()
    test = df[df["period_dt"] >= "2025-01-01"].copy()
    return train, validation, test


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 1e-6)

    absolute_error = np.abs(predicted - actual)
    absolute_log_error = np.abs(np.log(predicted) - np.log(actual))

    return {
        "mae_bhd": float(np.mean(absolute_error)),
        "median_ae_bhd": float(np.median(absolute_error)),
        "median_ape": float(np.median(absolute_error / actual)),
        "rmsle": float(
            np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
        ),
        "median_factor_error": float(np.exp(np.median(absolute_log_error))),
        "within_2x": float(np.mean((predicted >= actual / 2) & (predicted <= actual * 2))),
        "within_1_5x": float(
            np.mean((predicted >= actual / 1.5) & (predicted <= actual * 1.5))
        ),
    }


def tune(model_builder, train, validation):
    best = None

    for alpha in RIDGE_ALPHAS:
        model = model_builder(alpha)
        model.fit(train, np.log1p(train["value_bhd_num"]))

        prediction = predict_bhd(model, validation)
        metrics = regression_metrics(validation["value_bhd_num"], prediction)

        if best is None or metrics["rmsle"] < best["metrics"]["rmsle"]:
            best = {
                "alpha": alpha,
                "model": model,
                "metrics": metrics,
            }

    return best


def evaluate(name, data, model_builder):
    train, validation, test = split_by_time(data)

    best = tune(model_builder, train, validation)
    test_prediction = predict_bhd(best["model"], test)
    test_metrics = regression_metrics(test["value_bhd_num"], test_prediction)

    return {
        "model": name,
        "selected_alpha": best["alpha"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        **test_metrics,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = load_modeling_data(TIER1_DATA)
    tier2 = prepare_tier2_features(load_modeling_data(TIER2_DATA))

    # Fair comparison: Tier 1 is evaluated only on the same tender rows
    # that have Tier 2 data.
    keys = set(zip(tier2["report_period"], tier2["tender_number"]))
    tier1_same = tier1[
        tier1.apply(
            lambda row: (row["report_period"], row["tender_number"]) in keys,
            axis=1,
        )
    ].copy()

    results = [
        evaluate("tier1_same_tenders", tier1_same, build_tier1_model),
        evaluate("tier2", tier2, build_tier2_model),
    ]

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
