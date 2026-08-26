# Presentation plan — 15 minutes

## 1. Cover
Government Tender Bid Predictor  
Student name  
September 2026

## 2. Agenda
- The bidding problem
- Data and approach
- Prediction model
- Results
- Demo
- Limitations and next steps

## 3. The idea
One sentence:
> Estimate the likely winning-bid region before a company submits its tender price.

Visual: tender document → model → predicted price range.

## 4. Business problem & impact
Show the trade-off:
- Too high → lose the tender.
- Too low → win but damage profitability.
- Goal → a more informed competitive range.

Add the required footer on this slide only.

## 5. Dataset
Big-picture numbers only:
- Bahrain government tender awards
- 2016–2026
- approximately 20,000 award rows
- modern tenders enriched with pre-bid Tender Board information

Avoid discussing PDF extraction engineering unless asked in Q&A.

## 6. How the model works
Non-technical explanation:
> The model learns patterns from previous tenders with similar authorities, subjects, scope, and tender scale.

Show:
Tier 1 → basic tender information  
Tier 2 → description + bond + duration + other pre-bid metadata

## 7. Why two tiers?
Explain fallback:
- Tier 2 when archive information is available
- Tier 1 when it is not

## 8. Evaluation
Explain chronological testing before showing numbers.

Final chart:
Tier 1 vs Tier 2 on the same held-out tenders.

Numbers are inserted only after the final P2 dataset is trained.

## 9. Demo
Open the app and enter one example tender.
Show:
- point estimate
- competitive range
- whether Tier 1 or Tier 2 was used

## 10. Project outcome
What exists by the end:
- historical tender dataset
- trained prediction models
- evaluation pipeline
- usable demo app

## 11. Limitations
- incomplete Tier 2 availability
- unusual one-off contracts
- changing market conditions
- prediction interval can still be broad
- model supports pricing decisions; it does not replace commercial judgment

## 12. Recommendations & future work
- complete high-value Tier 2 enrichment
- retrain as new awards arrive
- monitor model drift
- test category-specific models
- production API / internal bidding dashboard

Close with the business value rather than a technical metric.
