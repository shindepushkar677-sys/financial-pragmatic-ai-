import statistics


INTENT_TO_SIGNAL = {
    "EXPANSION": "growth",
    "GENERAL_UPDATE": "neutral",
    "STRATEGIC_PROBING": "neutral",
    "COST_PRESSURE": "risk",
}

SIGNAL_TO_VALUE = {"growth": -1.0, "neutral": 0.0, "risk": 1.0}
INTENT_TO_SCORE = {
    "EXPANSION": 1.0,
    "COST_PRESSURE": -1.0,
    "GENERAL_UPDATE": 0.0,
    "STRATEGIC_PROBING": 0.2,
}


def compute_risk_score(intents):
    if not intents:
        print("INTENTS:", intents)
        print("SCORE:", 0.0)
        return 0.0

    score_sum = 0.0
    mapped_intents = []
    for item in intents:
        intent = str(item.get("intent", "GENERAL_UPDATE")).upper()
        value = INTENT_TO_SCORE.get(intent, 0.0)
        mapped_intents.append({"intent": intent, "value": value})
        score_sum += value

    score = score_sum / float(len(intents))
    print("INTENTS:", mapped_intents)
    print("SCORE:", score)
    return float(score)


def derive_signal(score):
    if score > 0.2:
        signal = "growth"
    elif score < -0.2:
        signal = "risk"
    else:
        signal = "neutral"

    print("FINAL SCORE:", score)
    print("FINAL SIGNAL:", signal)
    return signal


def derive_market_prediction(score):
    if score >= 65:
        return "DOWN"
    elif score <= 35:
        return "UP"
    else:
        return "NEUTRAL"


def generate_insight(score, intents):
    counts = {}
    total = len(intents)

    for item in intents:
        counts[item["intent"]] = counts.get(item["intent"], 0) + 1

    expansion = counts.get("EXPANSION", 0)
    cost = counts.get("COST_PRESSURE", 0)
    probing = counts.get("STRATEGIC_PROBING", 0)

    if total == 0:
        return "No interpretable signals were detected in the discussion."

    if score <= 35:
        if expansion >= cost:
            return (
                "Strong growth signals driven by expansion-heavy management commentary "
                "with limited cost pressure signals."
            )
        return "Growth-oriented messaging appears present but mixed with cost-related caution."

    if score >= 65:
        if cost > expansion:
            return (
                "Elevated risk profile led by persistent cost and margin pressure signals "
                "across management commentary."
            )
        return "Risk signals dominate this transcript despite intermittent growth commentary."

    if probing > 0 and cost > 0:
        return "Mixed outlook: analyst probing and cost pressure are balancing growth narratives."

    return "Balanced management commentary with no single dominant directional signal."


def compute_confidence(intents):
    counts = {"growth": 0, "neutral": 0, "risk": 0}
    total = len(intents)

    if total == 0:
        return 0.0

    for item in intents:
        signal = INTENT_TO_SIGNAL.get(item["intent"], "neutral")
        counts[signal] = counts.get(signal, 0) + 1

    dominant = max(counts.values())
    confidence = dominant / total

    return round(confidence * 100, 2)


def compute_signal_std(intents):
    if len(intents) <= 1:
        return 0.0

    series = []
    for item in intents:
        signal = INTENT_TO_SIGNAL.get(item["intent"], "neutral")
        series.append(SIGNAL_TO_VALUE[signal])

    return float(statistics.pstdev(series))


def detect_volatility(intents):
    if len(intents) <= 1:
        return "LOW"

    volatility_std = compute_signal_std(intents)
    if volatility_std >= 0.75:
        return "HIGH"
    elif volatility_std >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"


def compute_intent_distribution(intents):
    total = len(intents)
    counts = {
        "EXPANSION": 0,
        "GENERAL_UPDATE": 0,
        "STRATEGIC_PROBING": 0,
        "COST_PRESSURE": 0,
    }

    for item in intents:
        intent = item.get("intent", "GENERAL_UPDATE")
        if intent in counts:
            counts[intent] += 1

    if total == 0:
        return {key: 0.0 for key in counts}

    return {key: round((value / total) * 100, 2) for key, value in counts.items()}


def compute_signal_distribution(intents):
    counts = {"growth": 0, "neutral": 0, "risk": 0}
    for item in intents:
        intent = str(item.get("intent", "GENERAL_UPDATE")).upper()
        signal = INTENT_TO_SIGNAL.get(intent, "neutral")
        counts[signal] += 1
    return counts
