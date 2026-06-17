from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer


analyzer = EarningsCallAnalyzer()

transcript = """
CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.
"""

result = analyzer.analyze(transcript)

print("Timeline signals:\n")
for item in result["timeline_signals"]:
    print(item)

print("\nDominant signal:\n")
print(result["aggregation"]["dominant_signal"])

print("\nAI insight:\n")
print(result["insight"])
