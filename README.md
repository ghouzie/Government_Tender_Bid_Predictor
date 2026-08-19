# Government_Tender_Bid_Predictor
A data science project that predicts the likely winning bid amount for Bahrain government tenders, using historical award data published by the Tender Board.

## The problem
Companies bidding on government tenders can't see competitors' prices before submitting, and for one-of-a-kind contracts there's no past benchmark to guide pricing. Bid too high and lose the contract, bid too low and win at a loss.

## What the model does
Given a new tender's features (sector, category, contract size), the model predicts a realistic winning bid range, learned from thousands of past tenders where both the description and the actual awarded price are known.

## Data
Bahrain Tender Board's monthly Awarded Tenders reports (public PDFs, 10 years of history), enriched where available with fuller tender descriptions from the Tender Board's archive.

https://tenderbidpredictor.streamlit.app/
