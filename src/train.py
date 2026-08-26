import json

import joblib
import numpy as np

from config import MODEL_DIR, TIER1_DATA, TIER2_DATA
from data import load_modeling_data, prepare_tier2_features
from model import build_tier1_model, build_tier2_model


def train_and_save(data, model, path, metadata):
    model.fit(data, np.log1p(data["value_bhd_num"]))
    joblib.dump({"model": model, "metadata": metadata}, path)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = load_modeling_data(TIER1_DATA)
    tier2 = prepare_tier2_features(load_modeling_data(TIER2_DATA))

    tier1_model = build_tier1_model(alpha=10.0)
    tier2_model = build_tier2_model(alpha=10.0)

    train_and_save(
        tier1,
        tier1_model,
        MODEL_DIR / "tier1_model.joblib",
        {"tier": 1, "target": "winning_bid_bhd"},
    )

    train_and_save(
        tier2,
        tier2_model,
        MODEL_DIR / "tier2_model.joblib",
        {"tier": 2, "target": "winning_bid_bhd"},
    )

    print("Saved Tier 1 and Tier 2 models in models/")


if __name__ == "__main__":
    main()
