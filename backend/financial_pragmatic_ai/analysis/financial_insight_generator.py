def generate_insight(signal):

    if signal == "risk":
        return "Conversation suggests potential margin or profitability risk."

    if signal == "growth":
        return "Management discussion indicates a growth-oriented strategy."

    return "Discussion appears operational with no strong financial signals."
