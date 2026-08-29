from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_FILE = ROOT / "data/raw/tier2_checkpoint.csv"
TIER1_FILE = ROOT / "data/raw/tier1_modeling_dataset.csv"
REVIEW_FILE = ROOT / "data/review_choices.csv"

FINAL_FILE = ROOT / "data/processed/tier2_final.csv"
MODEL_FILE = ROOT / "data/processed/tier2_modeling.csv"
HISTORY_FILE = ROOT / "data/processed/historical_awards.csv"


def read_money(value):
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", str(value))

    if match is None:
        return np.nan

    return float(match.group(1).replace(",", ""))


def read_number(value):
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))

    if match is None:
        return np.nan

    return float(match.group(1))


def duration_to_months(value):
    text = str(value).lower()

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(year|month|week|day)",
        text,
    )

    if match is None:
        return np.nan

    amount = float(match.group(1))
    unit = match.group(2)

    if unit == "year":
        return amount * 12

    if unit == "month":
        return amount

    if unit == "week":
        return amount / 4.345

    return amount / 30.44


def choose_reviewed_rows(checkpoint, review):
    rows = []

    # Start with the automatic exact matches.
    exact = checkpoint[
        checkpoint["tier2_status"] == "accepted_exact"
    ].copy()

    excluded = set(
        review.loc[
            review["use_match"] == "no",
            "reference",
        ]
    )

    exact = exact[
        ~exact["tender_number"].isin(excluded)
    ]

    for _, row in exact.iterrows():
        rows.append(row)

    # Add manually reviewed multiple-match cases.
    approved = review[
        review["use_match"] == "yes"
    ]

    for _, choice in approved.iterrows():
        reference = choice["reference"]
        selected_page = choice["selected_tender_page"]

        matches = checkpoint[
            (checkpoint["tender_number"] == reference)
            & checkpoint["tier2_status"].str.startswith("review")
        ]

        if matches.empty:
            continue

        selected = matches[
            matches["tier2_tender_number_full"].str.contains(
                re.escape(selected_page),
                regex=True,
                na=False,
            )
        ]

        if not selected.empty:
            rows.append(selected.iloc[0])

    result = pd.DataFrame(rows)

    # One Tender Board page per normalized PA reference.
    result = result.drop_duplicates(
        "tender_number_normalized",
        keep="first",
    )

    return result


def choose_award_row(tier1, reference, closing_date):
    rows = tier1[
        tier1["tender_number_normalized"] == reference
    ].copy()

    if rows.empty:
        return None

    rows["award_date"] = (
        pd.to_datetime(
            rows["report_period"] + "-01",
            errors="coerce",
        )
        + pd.offsets.MonthEnd(0)
    )

    close = pd.to_datetime(
        closing_date,
        errors="coerce",
    )

    if pd.isna(close):
        return rows.sort_values("award_date").iloc[-1]

    rows["days_after_close"] = (
        rows["award_date"] - close
    ).dt.days

    after_close = rows[
        rows["days_after_close"] >= 0
    ]

    if not after_close.empty:
        return after_close.sort_values(
            "days_after_close"
        ).iloc[0]

    rows["date_distance"] = (
        rows["days_after_close"].abs()
    )

    return rows.sort_values(
        "date_distance"
    ).iloc[0]


def build_tier2_dataset(tier1, tender_pages):
    output = []

    for _, tender in tender_pages.iterrows():
        award = choose_award_row(
            tier1,
            tender["tender_number_normalized"],
            tender["tier2_closing_date"],
        )

        if award is None:
            continue

        publish_date = pd.to_datetime(
            tender["tier2_publish_date"],
            errors="coerce",
        )

        closing_date = pd.to_datetime(
            tender["tier2_closing_date"],
            errors="coerce",
        )

        days_to_close = np.nan

        if (
            not pd.isna(publish_date)
            and not pd.isna(closing_date)
        ):
            days_to_close = (
                closing_date - publish_date
            ).days

        description = (
            str(tender["tier2_description"]).strip()
            + " "
            + str(tender["tier2_additional_notes"]).strip()
        ).strip()

        output.append({
            "reference": award["tender_number"],
            "report_period": award["report_period"],
            "year": int(award["report_period"][:4]),
            "entity": award["entity_feature"],
            "sector": award["sector_feature"],
            "subject": award["subject_feature"],
            "description": description,
            "issued_by": (
                tender["tier2_issued_by"]
                or "UNKNOWN"
            ),
            "internal_external": (
                tender["tier2_internal_external"]
                or "UNKNOWN"
            ),
            "invitation_method": (
                tender["tier2_invitation_method"]
                or "UNKNOWN"
            ),
            "alternate_bid": (
                tender["tier2_alternate_bid_allowed"]
                or "UNKNOWN"
            ),
            "initial_bond": read_money(
                tender["tier2_initial_bond"]
            ),
            "tender_fee": read_money(
                tender["tier2_tender_fees"]
            ),
            "bond_validity_days": read_number(
                tender["tier2_bond_validity"]
            ),
            "contract_duration_months": duration_to_months(
                tender["tier2_contract_duration"]
            ),
            "days_to_close": days_to_close,
            "publish_year": (
                publish_date.year
                if not pd.isna(publish_date)
                else np.nan
            ),
            "publish_month": (
                publish_date.month
                if not pd.isna(publish_date)
                else np.nan
            ),
            "winning_bid_bhd": float(
                award["value_bhd"]
            ),
            "priority": tender["enrichment_priority"],
            "source_url": tender["tier2_source_url"],
        })

    return pd.DataFrame(output)


def build_history_file(tier1):
    history = tier1.copy()

    history["year"] = (
        history["report_period"]
        .str[:4]
        .astype(int)
    )

    history["winning_bid_bhd"] = pd.to_numeric(
        history["value_bhd"],
        errors="coerce",
    )

    history = history[
        history["winning_bid_bhd"] > 0
    ][[
        "tender_number",
        "report_period",
        "year",
        "entity_feature",
        "sector_feature",
        "subject_feature",
        "winning_bid_bhd",
    ]]

    history.columns = [
        "reference",
        "report_period",
        "year",
        "entity",
        "sector",
        "subject",
        "winning_bid_bhd",
    ]

    return history


def main():
    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        dtype=str,
    ).fillna("")

    tier1 = pd.read_csv(
        TIER1_FILE,
        dtype=str,
    ).fillna("")

    review = pd.read_csv(
        REVIEW_FILE,
        dtype=str,
    ).fillna("")

    tender_pages = choose_reviewed_rows(
        checkpoint,
        review,
    )

    final_data = build_tier2_dataset(
        tier1,
        tender_pages,
    )

    final_data.to_csv(
        FINAL_FILE,
        index=False,
    )

    # Some award reports contain a unit/tariff value instead of
    # a full contract value. If the award is lower than the
    # required bond, it is not comparable to the other targets.
    bad_price = (
        final_data["initial_bond"].notna()
        & (
            final_data["winning_bid_bhd"]
            < final_data["initial_bond"]
        )
    )

    modeling_data = final_data[
        ~bad_price
    ].copy()

    modeling_data.to_csv(
        MODEL_FILE,
        index=False,
    )

    history = build_history_file(tier1)

    history.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print("Clean enriched tenders:", len(final_data))
    print("Rows used for modeling:", len(modeling_data))
    print("Historical award rows:", len(history))


if __name__ == "__main__":
    main()
