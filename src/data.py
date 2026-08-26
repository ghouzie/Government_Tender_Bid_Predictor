import re

import numpy as np
import pandas as pd


def load_modeling_data(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df["period_dt"] = pd.to_datetime(df["report_period"] + "-01", errors="coerce")

    value_column = "value_bhd"
    if "value_bhd_tier1" in df.columns and df[value_column].eq("").any():
        df[value_column] = df[value_column].where(
            df[value_column].ne(""), df["value_bhd_tier1"]
        )

    df["value_bhd_num"] = pd.to_numeric(df[value_column], errors="coerce")
    return df[df["value_bhd_num"] > 0].copy()


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_money(value):
    match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", str(value or ""))
    if not match:
        return np.nan
    return float(match.group(1).replace(",", ""))


def parse_number(value):
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    return float(match.group(1)) if match else np.nan


def duration_to_months(value):
    text = str(value or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", text)
    if not match:
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


def _fill_feature(df, column, fallbacks, missing_value="__MISSING__"):
    if column not in df.columns:
        df[column] = ""

    result = df[column].copy()
    for fallback in fallbacks:
        if fallback in df.columns:
            result = result.where(result.ne(""), df[fallback])

    return result.replace("", missing_value)


def prepare_tier2_features(df):
    df = df.copy()

    df["entity_feature"] = _fill_feature(
        df,
        "entity_feature",
        ["entity_feature_tier1", "entity_normalized", "entity"],
    )
    df["sector_feature"] = _fill_feature(
        df,
        "sector_feature",
        ["sector_feature_tier1", "sector_normalized", "sector"],
    )
    df["subject_feature"] = _fill_feature(
        df,
        "subject_feature",
        ["subject_feature_tier1", "subject"],
        missing_value="",
    ).map(clean_text)

    description = df.get("tier2_description", pd.Series("", index=df.index))
    notes = df.get("tier2_additional_notes", pd.Series("", index=df.index))
    df["tier2_text"] = (
        description.map(clean_text) + " " + notes.map(clean_text)
    ).str.strip()

    for source, target in [
        ("tier2_issued_by", "issued_by"),
        ("tier2_internal_external", "internal_external"),
        ("tier2_invitation_method", "invitation_method"),
        ("tier2_alternate_bid_allowed", "alternate_bid"),
    ]:
        values = df[source] if source in df.columns else pd.Series("", index=df.index)
        df[target] = values.replace("", "UNKNOWN")

    def series_or_blank(column):
        if column in df.columns:
            return df[column]
        return pd.Series("", index=df.index)

    df["bond_bhd"] = series_or_blank("tier2_initial_bond").map(parse_money)
    df["tender_fee_bhd"] = series_or_blank("tier2_tender_fees").map(parse_money)
    df["bond_validity_days"] = series_or_blank("tier2_bond_validity").map(parse_number)
    df["contract_duration_months"] = series_or_blank(
        "tier2_contract_duration"
    ).map(duration_to_months)

    publish = pd.to_datetime(series_or_blank("tier2_publish_date"), errors="coerce")
    closing = pd.to_datetime(series_or_blank("tier2_closing_date"), errors="coerce")
    df["days_publish_to_close"] = (closing - publish).dt.days.astype(float)

    numeric = [
        "bond_bhd",
        "tender_fee_bhd",
        "bond_validity_days",
        "contract_duration_months",
        "days_publish_to_close",
    ]

    for column in numeric:
        df[f"log_{column}"] = np.log1p(df[column].clip(lower=0))

    return df
