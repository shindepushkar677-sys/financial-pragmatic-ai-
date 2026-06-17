def predict_market_outlook(signal, risk_score, volatility, intent_distribution):
    expansion_share = intent_distribution.get("EXPANSION", 0.0)
    risk_share = (
        intent_distribution.get("COST_PRESSURE", 0.0)
        + intent_distribution.get("STRATEGIC_PROBING", 0.0)
    )

    if signal == "growth" and volatility == "LOW" and risk_score <= 45:
        return {
            "prediction": "UP",
            "explanation": (
                "Prediction is UP due to strong expansion signals and low volatility."
            ),
        }

    if signal == "risk" and volatility == "HIGH" and risk_score >= 60:
        return {
            "prediction": "DOWN",
            "explanation": (
                "Prediction is DOWN due to persistent risk signals and high volatility."
            ),
        }

    if signal == "neutral" and volatility == "HIGH":
        return {
            "prediction": "VOLATILE",
            "explanation": (
                "Prediction is VOLATILE because mixed signals are accompanied by elevated volatility."
            ),
        }

    if risk_share > expansion_share and risk_score >= 55:
        return {
            "prediction": "DOWN",
            "explanation": (
                "Prediction is DOWN as risk-weighted intents dominate the conversation."
            ),
        }

    if expansion_share > risk_share and risk_score <= 45:
        return {
            "prediction": "UP",
            "explanation": (
                "Prediction is UP as expansion signals lead with manageable risk pressure."
            ),
        }

    return {
        "prediction": "NEUTRAL",
        "explanation": (
            "Prediction is NEUTRAL due to balanced signals without a strong directional edge."
        ),
    }
