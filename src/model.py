import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TIER1_CATEGORICAL = ["entity_feature", "sector_feature"]

TIER2_CATEGORICAL = [
    "entity_feature",
    "sector_feature",
    "issued_by",
    "internal_external",
    "invitation_method",
    "alternate_bid",
]

TIER2_NUMERIC = [
    "log_bond_bhd",
    "log_tender_fee_bhd",
    "log_bond_validity_days",
    "log_contract_duration_months",
    "log_days_publish_to_close",
]


def build_tier1_model(alpha=10.0):
    features = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                TIER1_CATEGORICAL,
            ),
            (
                "subject",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
                "subject_feature",
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def build_tier2_model(alpha=10.0):
    features = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                TIER2_CATEGORICAL,
            ),
            (
                "subject",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=18000,
                    sublinear_tf=True,
                ),
                "subject_feature",
            ),
            (
                "scope",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=28000,
                    sublinear_tf=True,
                ),
                "tier2_text",
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                TIER2_NUMERIC,
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("model", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


def predict_bhd(model, data):
    prediction = np.expm1(model.predict(data))
    return np.maximum(prediction, 0)
