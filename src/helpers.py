import numpy as np


def regression_metrics(actual, predicted):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.maximum(
        np.asarray(predicted, dtype=float),
        0.01,
    )

    percentage_error = (
        np.abs(predicted - actual)
        / actual
    )

    log_error = (
        np.log1p(predicted)
        - np.log1p(actual)
    )

    return {
        "median_percentage_error": float(
            np.median(percentage_error)
        ),
        "rmsle": float(
            np.sqrt(
                np.mean(log_error ** 2)
            )
        ),
        "within_2x": float(
            np.mean(
                (predicted >= actual / 2)
                & (predicted <= actual * 2)
            )
        ),
        "within_1_5x": float(
            np.mean(
                (predicted >= actual / 1.5)
                & (predicted <= actual * 1.5)
            )
        ),
    }
