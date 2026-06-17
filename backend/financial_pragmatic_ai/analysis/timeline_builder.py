def build_timeline(segments):
    timeline = []

    for i, seg in enumerate(segments):
        timeline.append(
            {
                "time": i,
                "speaker": seg["speaker"],
                "intent": seg["intent"],
            }
        )

    return timeline
