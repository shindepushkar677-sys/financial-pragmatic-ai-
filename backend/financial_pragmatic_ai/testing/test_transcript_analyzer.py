from financial_pragmatic_ai.analysis.transcript_analyzer import TranscriptAnalyzer
from financial_pragmatic_ai.analysis.financial_insight_generator import generate_insight

analyzer = TranscriptAnalyzer()

transcript = """
CEO: We plan to expand operations in Asia next quarter.
CFO: Margins may compress due to supply chain costs.
Analyst: How will this impact profitability going forward?
"""

analysis = analyzer.analyze(transcript)

segments = analysis
signal = analyzer.predict_conversation_signal(segments)

print("\nSegments:\n")

for seg in segments:
    print(seg)

print("\nConversation signal:\n")
print(signal)

insight = generate_insight(signal)

print("\nAI Insight:\n")
print(insight)
