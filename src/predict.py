from pathlib import Path

import joblib
import pandas as pd

from data import prepare_tier2_features
from model import predict_bhd


class TenderBidPredictor:
    def __init__(self, tier1_path, tier2_path=None):
        self.tier1 = joblib.load(tier1_path)
        self.tier2 = joblib.load(tier2_path) if tier2_path and Path(tier2_path).exists() else None

    def predict_tier1(self, entity, sector, subject):
        row = pd.DataFrame(
            [
                {
                    "entity_feature": entity or "__MISSING__",
                    "sector_feature": sector or "__MISSING__",
                    "subject_feature": subject or "",
                }
            ]
        )
        return float(predict_bhd(self.tier1["model"], row)[0])

    def predict_tier2(self, values):
        if self.tier2 is None:
            raise RuntimeError("Tier 2 model has not been trained yet.")

        row = pd.DataFrame([values])
        row = prepare_tier2_features(row)
        return float(predict_bhd(self.tier2["model"], row)[0])
