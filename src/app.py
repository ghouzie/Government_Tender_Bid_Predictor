import streamlit as st

from config import TIER1_MODEL, TIER2_MODEL
from predict import TenderBidPredictor


st.set_page_config(
    page_title="Government Tender Bid Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("Government Tender Bid Predictor")
st.caption("Estimate a competitive winning-bid range from historical Bahrain tenders.")

if not TIER1_MODEL.exists():
    st.warning(
        "Model files have not been trained yet. "
        "Run `python src/train.py` after the final processed datasets are ready."
    )
    st.stop()

predictor = TenderBidPredictor(TIER1_MODEL, TIER2_MODEL)

st.subheader("Tender details")

entity = st.text_input("Issuing authority")
sector = st.text_input("Sector")
subject = st.text_area("Tender subject", height=90)

with st.expander("Additional tender information (Tier 2)"):
    description = st.text_area("Description / scope", height=120)
    issued_by = st.text_input("Issued by")
    internal_external = st.selectbox(
        "Internal / External",
        ["", "External", "Internal"],
    )
    invitation_method = st.text_input("Invitation method")
    bond = st.number_input("Initial bond (BHD)", min_value=0.0, step=100.0)
    fee = st.number_input("Tender fee (BHD)", min_value=0.0, step=10.0)
    validity = st.number_input("Bid validity (days)", min_value=0, step=1)
    duration = st.text_input("Contract duration", placeholder="Example: 2 Year")
    alternate = st.selectbox("Alternate bid allowed", ["", "Yes", "No"])
    publish_date = st.date_input("Publish date", value=None)
    closing_date = st.date_input("Closing date", value=None)

if st.button("Estimate winning bid", type="primary"):
    if not subject.strip():
        st.error("Enter at least a tender subject.")
        st.stop()

    has_tier2 = predictor.tier2 is not None and (
        description.strip()
        or bond > 0
        or duration.strip()
        or invitation_method.strip()
    )

    if has_tier2:
        point = predictor.predict_tier2(
            {
                "entity_feature": entity or "__MISSING__",
                "sector_feature": sector or "__MISSING__",
                "subject_feature": subject,
                "tier2_description": description,
                "tier2_additional_notes": "",
                "tier2_issued_by": issued_by,
                "tier2_internal_external": internal_external,
                "tier2_invitation_method": invitation_method,
                "tier2_initial_bond": str(bond) if bond else "",
                "tier2_tender_fees": str(fee) if fee else "",
                "tier2_bond_validity": str(validity) if validity else "",
                "tier2_contract_duration": duration,
                "tier2_alternate_bid_allowed": alternate,
                "tier2_publish_date": str(publish_date) if publish_date else "",
                "tier2_closing_date": str(closing_date) if closing_date else "",
            }
        )
        tier = "Tier 2"
    else:
        point = predictor.predict_tier1(entity, sector, subject)
        tier = "Tier 1"

    # This will be replaced by the final calibrated interval after P2 training.
    range_factor = 2.4
    lower = point / range_factor
    upper = point * range_factor

    st.subheader("Estimated competitive range")
    st.metric("Point estimate", f"BHD {point:,.0f}")
    st.write(f"**Range:** BHD {lower:,.0f} – {upper:,.0f}")
    st.caption(f"Prediction used: {tier}")
