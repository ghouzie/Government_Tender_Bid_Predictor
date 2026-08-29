from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "data/processed/tier2_modeling.csv"
HISTORY_FILE = ROOT / "data/processed/historical_awards.csv"
OUTPUT_FILE = ROOT / "data/processed/tier2_with_history.csv"


def make_ridge_model():
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

    model = Ridge(
        alpha=3,
        solver="lsqr",
    )

    return preprocessor, model


def historical_ridge_prediction(history, new_rows):
    preprocessor, model = make_ridge_model()

    x_history = preprocessor.fit_transform(history)

    model.fit(
        x_history,
        np.log1p(
            history["winning_bid_bhd"]
        ),
    )

    x_new = preprocessor.transform(new_rows)

    prediction = np.expm1(
        model.predict(x_new)
    )

    return prediction


def historical_similarity_price(history, new_rows):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
    )

    old_vectors = vectorizer.fit_transform(
        history["subject"].fillna("")
    )

    new_vectors = vectorizer.transform(
        new_rows["subject"].fillna("")
    )

    similarity = new_vectors @ old_vectors.T

    old_prices = history[
        "winning_bid_bhd"
    ].to_numpy()

    prices = []
    top_scores = []

    for row_number in range(
        similarity.shape[0]
    ):
        row = similarity.getrow(row_number)

        if row.nnz == 0:
            prices.append(np.nan)
            top_scores.append(0.0)
            continue

        order = np.argsort(
            row.data
        )[-5:][::-1]

        indexes = row.indices[order]
        scores = row.data[order]

        weights = scores / scores.sum()

        # Geometric average because tender values vary greatly.
        log_price = np.sum(
            weights
            * np.log1p(
                old_prices[indexes]
            )
        )

        prices.append(
            np.expm1(log_price)
        )

        top_scores.append(
            float(scores[0])
        )

    return prices, top_scores


def main():
    data = pd.read_csv(MODEL_FILE)
    history = pd.read_csv(HISTORY_FILE)

    data["history_ridge"] = np.nan
    data["history_similar_price"] = np.nan
    data["history_similarity"] = np.nan

    # Each year's history feature only uses older tenders.
    # For example, 2025 rows use award history up to 2024.
    for year in sorted(data["year"].unique()):
        older = history[
            history["year"] < year
        ].copy()

        current = data[
            data["year"] == year
        ].copy()

        if older.empty or current.empty:
            continue

        ridge_prediction = (
            historical_ridge_prediction(
                older,
                current,
            )
        )

        similar_price, similarity = (
            historical_similarity_price(
                older,
                current,
            )
        )

        data.loc[
            current.index,
            "history_ridge",
        ] = ridge_prediction

        data.loc[
            current.index,
            "history_similar_price",
        ] = similar_price

        data.loc[
            current.index,
            "history_similarity",
        ] = similarity

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Saved rows with historical features:",
        len(data),
    )


if __name__ == "__main__":
    main()
