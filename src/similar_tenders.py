import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def find_history_matches(
    history,
    subject,
    max_year=2024,
    number=5,
):
    """Used for the model's 2016+ historical price feature."""
    older = history[
        history["year"] <= max_year
    ].copy()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )

    old_vectors = vectorizer.fit_transform(
        older["subject"].fillna("")
    )

    new_vector = vectorizer.transform(
        [subject]
    )

    similarity = new_vector @ old_vectors.T
    scores = similarity.toarray()[0]

    best = np.argsort(
        scores
    )[::-1][:number]

    result = older.iloc[
        best
    ].copy()

    result["similarity"] = scores[best]

    return result


def similar_price(matches):
    if matches.empty:
        return np.nan

    scores = matches[
        "similarity"
    ].to_numpy()

    prices = matches[
        "winning_bid_bhd"
    ].to_numpy()

    if scores.sum() == 0:
        return np.nan

    weights = scores / scores.sum()

    log_price = np.sum(
        weights
        * np.log1p(prices)
    )

    return float(
        np.expm1(log_price)
    )


def price_proximity(values, target):
    values = np.asarray(
        values,
        dtype=float,
    )

    if target is None or target <= 0:
        return np.zeros(
            len(values)
        )

    return np.exp(
        -np.abs(
            np.log1p(values)
            - np.log1p(target)
        )
    )


def find_similar_tenders(
    enriched,
    subject,
    description="",
    sector="",
    entity="",
    initial_bond=0,
    duration=0,
    max_year=2024,
    number=3,
):
    """Richer historical comparisons shown on the dashboard."""
    older = enriched[
        enriched["year"] <= max_year
    ].copy()

    old_text = (
        older["subject"].fillna("")
        + " "
        + older["description"].fillna("")
    )

    new_text = (
        str(subject)
        + " "
        + str(description)
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=15000,
        sublinear_tf=True,
    )

    old_vectors = vectorizer.fit_transform(
        old_text
    )

    new_vector = vectorizer.transform(
        [new_text]
    )

    text_similarity = (
        new_vector @ old_vectors.T
    ).toarray()[0]

    same_sector = (
        older["sector"].fillna("")
        == sector
    ).astype(float).to_numpy()

    same_entity = (
        older["entity"].fillna("")
        == entity
    ).astype(float).to_numpy()

    bond_similarity = price_proximity(
        older["initial_bond"].fillna(0),
        initial_bond,
    )

    duration_similarity = price_proximity(
        older[
            "contract_duration_months"
        ].fillna(0),
        duration,
    )

    score = (
        0.65 * text_similarity
        + 0.10 * same_sector
        + 0.05 * same_entity
        + 0.10 * bond_similarity
        + 0.10 * duration_similarity
    )

    best = np.argsort(
        score
    )[::-1][:number]

    result = older.iloc[
        best
    ].copy()

    result["text_similarity"] = (
        text_similarity[best]
    )

    result["match_score"] = score[best]

    return result


def input_quality(
    history_matches,
    filled_fields,
):
    if history_matches.empty:
        return "Low"

    text_similarity = float(
        history_matches.iloc[0][
            "similarity"
        ]
    )

    if text_similarity < 0.18:
        return "Low"

    if (
        text_similarity >= 0.30
        and filled_fields >= 3
    ):
        return "High"

    return "Moderate"
