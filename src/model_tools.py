import numpy as np

from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORY_COLUMNS = [
    "entity",
    "sector",
    "issued_by",
    "internal_external",
    "invitation_method",
    "alternate_bid",
]

NUMBER_COLUMNS = [
    "log_initial_bond",
    "log_tender_fee",
    "log_bond_validity_days",
    "log_contract_duration_months",
    "log_days_to_close",
    "publish_year",
    "publish_month",
    "log_history_ridge",
    "log_history_similar_price",
    "history_similarity",
]


def add_log_columns(data):
    data = data.copy()

    columns = [
        "initial_bond",
        "tender_fee",
        "bond_validity_days",
        "contract_duration_months",
        "days_to_close",
        "history_ridge",
        "history_similar_price",
    ]

    for column in columns:
        values = data[column].clip(lower=0)
        data["log_" + column] = np.log1p(values)

    return data


def make_preprocessor():
    return ColumnTransformer([
        (
            "categories",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
            ),
            CATEGORY_COLUMNS,
        ),
        (
            "subject",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=3000,
                sublinear_tf=True,
            ),
            "subject",
        ),
        (
            "description",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_features=5000,
                sublinear_tf=True,
            ),
            "description",
        ),
        (
            "numbers",
            Pipeline([
                (
                    "fill_missing",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scale",
                    StandardScaler(
                        with_mean=False,
                    ),
                ),
            ]),
            NUMBER_COLUMNS,
        ),
    ])


def make_lightgbm():
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=450,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=15,
        reg_alpha=2,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def to_bhd(log_prediction):
    return np.maximum(
        np.expm1(log_prediction),
        0,
    )
