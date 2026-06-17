import pandas as pd
import itertools
import random

INTENTS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE"
]

rows = []

for ceo, cfo, analyst in itertools.product(INTENTS, repeat=3):

    risk_score = 0
    growth_score = 0

    # CFO cost pressure increases risk
    if cfo == "COST_PRESSURE":
        risk_score += 0.6

    # Analyst probing increases risk
    if analyst == "STRATEGIC_PROBING":
        risk_score += 0.3

    # CEO expansion increases growth
    if ceo == "EXPANSION":
        growth_score += 0.6

    # CEO expansion + CFO pressure creates mixed signal
    if ceo == "EXPANSION" and cfo == "COST_PRESSURE":
        risk_score += 0.2

    # normalize

    neutral_score = max(0, 1 - (risk_score + growth_score))

    probs = {
        "risk": risk_score,
        "growth": growth_score,
        "neutral": neutral_score
    }

    signal = max(probs, key=probs.get)

    rows.append({
        "CEO_intent": ceo,
        "CFO_intent": cfo,
        "Analyst_intent": analyst,
        "signal": signal
    })

df = pd.DataFrame(rows)

df.to_csv(
    "financial_pragmatic_ai/data/conversation_signal_dataset.csv",
    index=False
)

print("Dataset built")
print(df.head())