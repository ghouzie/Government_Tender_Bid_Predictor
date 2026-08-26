# Government Tender Bid Predictor

A machine-learning project that estimates a competitive winning-bid range for Bahrain government tenders using historical award data and information available before bidding.

## Problem

When a company prepares a government tender bid, it usually cannot see competitors' prices before submission. A bid that is too high may lose the tender, while a bid that is too low can win at an unprofitable price.

This project uses historical Bahrain tender awards to estimate the likely winning-bid region for a new tender.

## Approach

The project uses two prediction tiers:

- **Tier 1:** issuing authority, sector, and tender subject.
- **Tier 2:** Tier 1 features plus pre-bid Tender Board information such as description, initial bond, contract duration, tender fee, invitation method, and bid validity.

If Tier 2 information is unavailable, the app can fall back to Tier 1.

## Dataset

The historical award dataset was compiled from Bahrain Tender Board awarded-tender reports covering 2016–2026.

Tier 2 records are enriched from Tender Board tender pages. Only information that would have been known before bid submission is used as model input.

Post-bid information such as competitor prices, bidder counts, and the winning bidder is excluded from prediction features.

> Final row counts and Tier 2 coverage will be updated after the current P2 enrichment run.

## Model

The current approach predicts the logarithm of the winning bid.

Categorical features are one-hot encoded, tender text is represented with character-level TF-IDF, and Ridge regression is used for the final prediction.

The evaluation uses chronological splits rather than a random split:

- training: up to 2023
- validation: 2024
- test: 2025+

This is intended to better represent the real use case of predicting future tenders from past records.

## Evaluation

Final metrics will be inserted after the expanded Tier 2 dataset is complete.

The main metrics are:

- Median Absolute Percentage Error
- RMSLE
- Median multiplicative error
- Percentage of predictions within 1.5× of the actual value
- Percentage of predictions within 2× of the actual value
- Coverage of the predicted bid range

## Run the project

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Place the processed datasets at:

```text
data/processed/tier1_modeling.csv
data/processed/tier2_modeling.csv
```

Evaluate the models:

```bash
python src/evaluate.py
```

Train the final artifacts:

```bash
python src/train.py
```

Run the demo:

```bash
streamlit run src/app.py
```

## Repository structure

```text
.
├── assets/
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
├── demo/
├── models/
├── outputs/
├── presentation/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Limitations

- Tender values vary over several orders of magnitude.
- Not every historical tender has Tier 2 information available.
- Some issuing authorities and tender categories have limited historical data.
- The model estimates a competitive region; it cannot know competitors' future pricing strategies.
- Inflation, policy changes, market conditions, and unusual one-off contracts can reduce accuracy.

## Future work

Insha'allah, future work can add more Tier 2 history, better uncertainty estimation, category-specific models, and monitoring for model drift as new tender awards become available.

## License

This capstone repository is intended for educational and portfolio use.
