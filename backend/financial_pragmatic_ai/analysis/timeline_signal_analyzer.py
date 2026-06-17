class TimelineSignalAnalyzer:

    def __init__(self, transcript_analyzer):
        self.transcript_analyzer = transcript_analyzer

    def analyze_timeline(self, segments):

        signals = []
        window_size = 3

        for i in range(len(segments) - window_size + 1):

            window = segments[i : i + window_size]

            signal = self.transcript_analyzer.predict_conversation_signal(window)

            signals.append({
                "window": window,
                "signal": signal
            })

        return signals
