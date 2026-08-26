# Demo video draft — 3 to 5 minutes

## 0:00–0:35 — Problem
Government tender bidders must choose a price before seeing competitors' bids.
Pricing too high can lose the tender; pricing too low can win at a poor margin.

## 0:35–0:55 — Solution
The project learns from historical Bahrain government tender awards and estimates a likely winning-bid range.

Mention the two tiers briefly:
- Tier 1 uses common tender information.
- Tier 2 adds richer pre-bid Tender Board information when available.

## 0:55–2:40 — Live app
1. Open the Streamlit app.
2. Enter authority, sector, and tender subject.
3. Show the Tier 1 estimate.
4. Add description, initial bond, duration, and invitation information.
5. Run again and show the Tier 2 estimate/range.

Use one prepared example so the recording stays smooth.

## 2:40–3:30 — Results
Show one simple comparison chart:
- Tier 1 baseline
- Tier 2 final model

Use the final held-out test metrics after P2 is complete.

## 3:30–4:00 — Lesson learned
Suggested point:
> The strongest improvement did not come from using more text alone. Structured pre-bid information, especially the initial bond and contract metadata, gave the model a much better signal of tender scale.

## Closing
The model is a decision-support tool. It gives bidders a historical benchmark, not a guarantee of the winning price.
