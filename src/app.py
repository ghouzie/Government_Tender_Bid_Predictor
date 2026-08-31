from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_tools import (
    add_log_columns,
    to_bhd,
)
from similar_tenders import (
    find_history_matches,
    find_similar_tenders,
    input_quality,
    similar_price,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models/final_model.joblib"
HISTORY_MODEL_FILE = ROOT / "models/history_model.joblib"

DATA_FILE = ROOT / "data/processed/tier2_with_history.csv"
HISTORY_FILE = ROOT / "data/processed/historical_awards.csv"


@st.cache_resource
def load_models():
    final_model = joblib.load(
        MODEL_FILE
    )

    history_model = joblib.load(
        HISTORY_MODEL_FILE
    )

    return final_model, history_model


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    history = pd.read_csv(HISTORY_FILE)

    return data, history


def historical_ridge(
    history_model,
    entity,
    sector,
    subject,
):
    today = datetime.now()

    row = pd.DataFrame([{
        "entity": entity,
        "sector": sector,
        "subject": subject,
    }])

    x = history_model[
        "preprocessor"
    ].transform(row)

    prediction = history_model[
        "model"
    ].predict(x)[0]

    return float(
        np.expm1(prediction)
    )


def predict_row(final_model, row):
    row = add_log_columns(row)

    x = final_model[
        "preprocessor"
    ].transform(row)

    prediction = final_model[
        "model"
    ].predict(x)

    return float(
        to_bhd(prediction)[0]
    )


def show_prediction(
    prediction,
    range_factor,
):
    lower = prediction / range_factor
    upper = prediction * range_factor

    st.metric(
        "Predicted winning bid",
        f"BHD {prediction:,.0f}",
    )

    st.write(
        "**80% prediction range:** "
        f"BHD {lower:,.0f} – "
        f"BHD {upper:,.0f}"
    )


def show_similar(matches, number=3):
    st.subheader("Similar historical tenders")

    st.caption(
        "These are shown as context. "
        "They are not copied as the model prediction."
    )

    for _, row in matches.head(number).iterrows():
        st.markdown(
            f"**{row['subject']}**  \n"
            f"{row['entity']} · {row['report_period']}  \n"
            f"Reference: `{row['reference']}`  \n"
            f"Historical award: "
            f"**BHD {row['winning_bid_bhd']:,.0f}**  \n"
            f"Comparison score: "
            f"**{row['match_score'] * 100:.0f}%**"
        )

        st.divider()


def new_tender_mode(
    final_model,
    history_model,
    data,
    history,
):
    st.subheader("New tender")

    entity = st.text_input(
        "Issuing authority"
    )

    sector = st.text_input(
        "Sector"
    )

    subject = st.text_area(
        "Tender subject"
    )

    description = st.text_area(
        "Description / scope"
    )

    col1, col2 = st.columns(2)

    with col1:
        initial_bond = st.number_input(
            "Initial bond (BHD)",
            min_value=0.0,
            step=100.0,
        )

        tender_fee = st.number_input(
            "Tender fee (BHD)",
            min_value=0.0,
            step=10.0,
        )

        duration = st.number_input(
            "Contract duration (months)",
            min_value=0.0,
            step=1.0,
        )

        validity = st.number_input(
            "Bond validity (days)",
            min_value=0.0,
            step=1.0,
        )

    with col2:
        invitation = st.text_input(
            "Invitation method"
        )

        internal_external = st.selectbox(
            "Internal / External",
            ["External", "Internal", "UNKNOWN"],
        )

        alternate_bid = st.selectbox(
            "Alternate bid allowed",
            ["No", "Yes", "UNKNOWN"],
        )

        days_to_close = st.number_input(
            "Days from publication to closing",
            min_value=0.0,
            step=1.0,
        )

    if not st.button(
        "Predict",
        type="primary",
    ):
        return

    if not subject.strip():
        st.error(
            "Please enter a tender subject."
        )
        return

    history_matches = find_history_matches(
        history,
        subject,
        max_year=2024,
        number=5,
    )

    display_matches = find_similar_tenders(
        data,
        subject=subject,
        description=description,
        sector=sector,
        entity=entity,
        initial_bond=initial_bond,
        duration=duration,
        max_year=2024,
        number=3,
    )

    filled_fields = sum([
        bool(description.strip()),
        initial_bond > 0,
        tender_fee > 0,
        duration > 0,
        bool(invitation.strip()),
    ])

    quality = input_quality(
        history_matches,
        filled_fields,
    )

    if quality == "Low":
        st.error(
            "This tender is too different from the "
            "historical data for a reliable prediction."
        )

        show_similar(
            display_matches,
            number=3,
        )

        return

    history_prediction = historical_ridge(
        history_model,
        entity,
        sector,
        subject,
    )

    history_price = similar_price(
        history_matches
    )
    today = datetime.now()
    row = pd.DataFrame([{
        "entity": entity,
        "sector": sector,
        "subject": subject,
        "description": description,
        "issued_by": entity,
        "internal_external": internal_external,
        "invitation_method": (
            invitation or "UNKNOWN"
        ),
        "alternate_bid": alternate_bid,
        "initial_bond": (
            initial_bond
            if initial_bond > 0
            else np.nan
        ),
        "tender_fee": (
            tender_fee
            if tender_fee > 0
            else np.nan
        ),
        "bond_validity_days": (
            validity
            if validity > 0
            else np.nan
        ),
        "contract_duration_months": (
            duration
            if duration > 0
            else np.nan
        ),
        "days_to_close": (
            days_to_close
            if days_to_close > 0
            else np.nan
        ),
        "publish_year": today.year,
        "publish_month": today.month,
        "history_ridge": history_prediction,
        "history_similar_price": history_price,
        "history_similarity": float(
            history_matches.iloc[0][
                "similarity"
            ]
        ),
    }])

    prediction = predict_row(
        final_model,
        row,
    )

    st.subheader("Prediction")

    show_prediction(
        prediction,
        final_model["range_factor"],
    )

    st.write(
        "**Input quality:**",
        quality,
    )

    st.write(
        "**Historical baseline:** "
        f"BHD {history_prediction:,.0f}"
    )

    show_similar(
        display_matches,
        number=3,
    )


def historical_mode(
    final_model,
    data,
    history,
):
    st.subheader(
        "Historical validation"
    )

    st.write(
        "Choose a 2025 tender that the model "
        "did not see during training."
    )

    test = data[
        data["year"] == 2025
    ].copy()

    test["label"] = (
        test["reference"]
        + " — "
        + test["subject"].str[:80]
    )

    default_rows = test[
        test["reference"] == "LF-233"
    ]

    default_index = 0

    if not default_rows.empty:
        default_label = (
            default_rows.iloc[0]["label"]
        )

        default_index = list(
            test["label"]
        ).index(default_label)

    selected_label = st.selectbox(
        "Historical tender",
        test["label"],
        index=default_index,
    )

    row = test[
        test["label"] == selected_label
    ].iloc[0]

    st.write(
        "**Authority:**",
        row["entity"],
    )

    st.write(
        "**Subject:**",
        row["subject"],
    )

    if st.button(
        "Run historical prediction",
        type="primary",
    ):
        st.session_state[
            "predicted_reference"
        ] = row["reference"]

        st.session_state[
            "revealed_reference"
        ] = ""

    if (
        st.session_state.get(
            "predicted_reference"
        )
        != row["reference"]
    ):
        return

    model_row = pd.DataFrame(
        [row.drop(labels=["label"])]
    )

    prediction = predict_row(
        final_model,
        model_row,
    )

    st.subheader("Model result")

    show_prediction(
        prediction,
        final_model["range_factor"],
    )

    matches = find_similar_tenders(
        data,
        subject=row["subject"],
        description=row["description"],
        sector=row["sector"],
        entity=row["entity"],
        initial_bond=row["initial_bond"],
        duration=row[
            "contract_duration_months"
        ],
        max_year=2024,
        number=3,
    )

    show_similar(
        matches,
        number=3,
    )

    if st.button(
        "Reveal historical award"
    ):
        st.session_state[
            "revealed_reference"
        ] = row["reference"]

    if (
        st.session_state.get(
            "revealed_reference"
        )
        == row["reference"]
    ):
        actual = float(
            row["winning_bid_bhd"]
        )

        error = (
            abs(prediction - actual)
            / actual
        )

        st.subheader(
            "Historical outcome"
        )

        st.metric(
            "Actual winning award",
            f"BHD {actual:,.0f}",
        )

        st.write(
            "**Prediction error:** "
            f"{error * 100:.1f}%"
        )

        st.caption(
            "The actual award is shown only after "
            "the model prediction. It is not a model input."
        )


def main():
    st.set_page_config(
        page_title=(
            "Government Tender Bid Predictor"
        ),
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "Government Tender Bid Predictor"
    )

    st.write(
        "Estimate a Bahrain government tender "
        "winning bid using information available "
        "before bidding."
    )

    final_model, history_model = (
        load_models()
    )

    data, history = load_data()

    mode = st.radio(
        "Mode",
        [
            "New Tender",
            "Historical Validation",
        ],
        horizontal=True,
    )

    if mode == "New Tender":
        new_tender_mode(
            final_model,
            history_model,
            data,
            history,
        )

    else:
        historical_mode(
            final_model,
            data,
            history,
        )


if __name__ == "__main__":
    main()
