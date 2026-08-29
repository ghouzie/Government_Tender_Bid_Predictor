# Government Tender Bid Predictor

A machine-learning capstone project that estimates the winning price of Bahrain government tenders using information available before bidding.

https://governmenttenderbidpredictor.streamlit.app/

## What the project does

The user enters information such as:

- issuing authority
- sector
- tender subject
- description / scope
- initial bond
- tender fee
- contract duration
- invitation method

The app returns:

- a point estimate in BHD
- an 80% prediction range
- an input-quality check
- three similar historical tenders

The app also has a **Historical Validation** mode. It lets you choose an unseen 2025 tender, run the model, then reveal the real historical award and the prediction error.

## Dataset pipeline

The dataset was not downloaded as a ready-made machine-learning dataset.

### 1. Award reports

Historical Bahrain Tender Board award reports were standardized into one award dataset covering 2016 onward.

The cleaned Tier 1 award dataset contains **8,847 rows**.

### 2. Archived tender enrichment

Recent award references were searched on the Tender Board Archived Tenders site.

The main join key was the normalized **PA reference number**.

Only:

- exact PA-reference matches
- or manually verified review cases

were accepted.

Ambiguous or incorrect matches were excluded.

### 3. Final enriched data

After the full P1-P3 enrichment:

- 1,899 clean enriched tenders were matched
- 6 unit/tariff-style award values were removed from price modeling
- 1,893 rows remain for the final price model

The six removed records had published award values lower than their required initial bond, showing that the value was not comparable to a full contract amount.

## Data leakage rule

A feature is only allowed if a bidder could know it before submitting a bid.

Used:

- tender subject
- description
- authority
- sector
- bond
- fee
- duration
- invitation information
- publication timing

Not used:

- competitor bid prices
- number of bidders
- winning bidder
- opened-bid results
- winning amount as an input

The winning amount is only the training target.

## Why 2016 data is still useful

Most rich Tier 2 archive fields are available only for newer tenders.

Older 2016+ awards are still used to create two historical benchmark features:

1. a Ridge estimate trained on earlier award history
2. a similarity-weighted historical price

For each year, these features are created using **older years only**.

For example:

- a 2024 tender uses history up to 2023
- a 2025 tender uses history up to 2024

This avoids future-data leakage.

## How TF-IDF is implemented

The model uses character-level TF-IDF for both the subject and description.

The code is in `src/model_tools.py`.

Example:

```python
TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=3000,
    sublinear_tf=True,
)
```

`analyzer="char_wb"` means the text is broken into character groups inside words.

For a word such as:

```text
maintenance
```

the vectorizer can learn patterns such as:

```text
mai
main
maint
aint
...
```

Character TF-IDF is useful here because the tender data contains Arabic, English, abbreviations and spelling variation.

`fit_transform()` is used on the training data to learn the TF-IDF vocabulary.

`transform()` is used for validation, test and new tenders. New data never creates a new vocabulary.

## Model comparison

Model choice is made using **2024 validation data**, not the 2025 test set.

Current validation comparison:

| Model | Median error |
|---|---:|
| LightGBM | 24.2% |
| Ridge + LightGBM | 24.9% |
| Linear SVR | 38.8% |
| Ridge | 39.5% |

LightGBM was selected because it had the lowest median percentage error.

The blend had slightly better RMSLE but worse median error, so the simpler single LightGBM model was kept.

## Why LightGBM

Ridge works well with sparse text features, but it assumes mostly linear relationships.

Tender prices have nonlinear interactions.

For example:

- a large initial bond
- plus a long contract
- plus a particular sector

can mean something different from any one field by itself.

LightGBM can learn these interactions with boosted decision trees.

## Chronological validation

The enriched model is evaluated chronologically:

- **2021-2023:** development training
- **2024:** validation and prediction-range calibration
- **2025:** untouched test set

The final evaluation model is then retrained using 2021-2024 and tested on 2025.

## Final 2025 results

Held-out 2025 rows: **373**

- Median percentage error: **25.1%**
- RMSLE: **0.627**
- Within 2x of actual: **86.6%**
- Within 1.5x of actual: **68.4%**

## Prediction range

The 80% prediction range is calibrated using 2024 validation errors only.

Current factor:

```text
1.762x
```

For prediction `P`:

```text
lower = P / 1.762
upper = P * 1.762
```

On the untouched 2025 set, the interval covered the actual award **81.2%** of the time.

## Similar historical tenders

The dashboard shows three comparable enriched tenders from 2024 or earlier.

The comparison score uses:

- subject and description text similarity
- same sector
- same authority
- similar initial bond
- similar contract duration

These comparisons are shown for context.

They are separate from the final LightGBM prediction.

## Project files

```text
src/
    prepare_data.py
    add_history_features.py
    model_tools.py
    compare_models.py
    train_model.py
    similar_tenders.py
    helpers.py
    app.py
```

### `prepare_data.py`

Builds the final cleaned enriched dataset from the Tender Board checkpoint.

### `add_history_features.py`

Uses 2016+ award history to create leakage-safe historical benchmark features.

### `compare_models.py`

Compares the candidate regression models on 2024.

### `train_model.py`

Trains the final LightGBM model, calibrates the range, tests on 2025 and saves the model artifacts.

### `similar_tenders.py`

Finds historical comparisons for the dashboard.

### `app.py`

Runs the Streamlit demo.

## Install

```bash
pip install -r requirements.txt
```

## Rebuild everything

```bash
python run_all.py
```

## Run the app

```bash
streamlit run src/app.py
```

## Limitations

- Tender prices can be affected by market conditions and bidder strategy that are not public before bidding.
- Some tender categories have limited historical examples.
- Similar historical tenders are contextual evidence, not proof of the future winning price.
- The prediction range can still be broad for unusual tenders.

## Future work

Insha'allah, future work can add more years of rich archive metadata, category-specific models and continuous retraining as new awards become available.
