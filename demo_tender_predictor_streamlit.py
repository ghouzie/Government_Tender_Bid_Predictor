
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Tender Bid Predictor - Demo", layout="centered")

st.title("Government Tender Bid Predictor")

st.markdown(
    "This shows the **historical average winning bid** for a given entity, "
    "built from real data extracted directly from the Tender Board's official "
    "Awarded Tenders report."
)

# 1. Load the real extracted data
df = pd.read_csv("tenders_v3.csv")

# 2. Clean: amount comes in as text with commas -> convert to a real number
df["amount_bhd"] = df["amount_bhd"].str.replace(",", "").astype(float)
df = df.dropna(subset=["amount_bhd", "entity"])
df = df[df["amount_bhd"] > 0]

col1, col2 = st.columns(2)
col1.metric("Clean rows used", len(df))
col2.metric("Amount range (BHD)", f"{df['amount_bhd'].min():,.0f} - {df['amount_bhd'].max():,.0f}")

with st.expander("See the underlying data"):
    st.dataframe(df[["entity", "tender_no", "winner", "amount_bhd"]])

# 3. Historical average winning bid by entity, computed directly from
#    the extracted data.
entity_avg = df.groupby("entity")["amount_bhd"].mean()

st.subheader("Try Predicting")

entity_choice = st.selectbox(
    "Pick a government entity (from this month's real data):",
    sorted(df["entity"].unique())
)

if st.button("Predict"):
    value = entity_avg[entity_choice]
    n = (df["entity"] == entity_choice).sum()
    st.info(f"Prediction of winning bid for this entity: **{value:,.0f} BHD** (from {n} tender{'s' if n != 1 else ''} in this dataset)")

st.divider()
st.caption(
    "This is a proof-of-concept on one month of data, using only one crude "
    "feature (which entity is buying). The real project scales this to 10 "
    "years of tenders with sector, category, and scope as features — this "
    "demo just shows the pipeline runs end to end."
)
