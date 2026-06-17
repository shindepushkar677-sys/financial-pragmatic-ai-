from collections import Counter


def compute_signal_stats(segments):
    intents = [s["intent"] for s in segments]
    counts = Counter(intents)
    return dict(counts)
