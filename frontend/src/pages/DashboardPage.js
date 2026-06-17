import { useMemo, useState } from "react";

import SummaryCard from "../components/SummaryCard";
import TimelineChart from "../components/TimelineChart";
import SignalHeatmap from "../components/SignalHeatmap";
import { analyzeTranscript, compareTranscripts, uploadTranscript } from "../api/client";

const SAMPLE = `CEO: We plan to expand operations globally.
CFO: Costs may rise due to supply chain issues.
Analyst: How will this impact margins?
CFO: We are monitoring cost structure carefully.`;

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [transcript, setTranscript] = useState(SAMPLE);
  const [file, setFile] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [compareTextA, setCompareTextA] = useState(SAMPLE);
  const [compareTextB, setCompareTextB] = useState(
    "CEO: Revenue increased strongly this quarter. CFO: Margins remained stable. Analyst: What is the outlook?"
  );
  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");

  const segments = result?.segments || [];

  const drivers = useMemo(() => {
    const growth = (result?.drivers?.growth_drivers || []).slice(0, 3);
    const risk = (result?.drivers?.risk_drivers || []).slice(0, 3);
    return { growth, risk };
  }, [result]);

  const signalDistribution = useMemo(() => {
    const order = ["EXPANSION", "GENERAL_UPDATE", "STRATEGIC_PROBING", "COST_PRESSURE"];
    const total = segments.length || 1;

    return order.map((intent) => {
      const count = segments.filter((segment) => segment.intent === intent).length;
      return { intent, count, percentage: Math.round((count / total) * 100) };
    });
  }, [segments]);

  const handleAnalyze = async () => {
    if (!transcript.trim()) {
      setError("Please enter a transcript");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const analysis = await analyzeTranscript(transcript);
      if (analysis?.error) {
        setError("Analysis failed. Try again.");
        setResult(null);
        return;
      }
      setResult(analysis);
    } catch (_err) {
      setError("Analysis failed. Try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a .txt or .pdf file");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const analysis = await uploadTranscript(file);
      if (analysis?.error) {
        setError("Upload failed. Try again.");
        setResult(null);
        return;
      }
      setResult(analysis);
      setFile(null);
      setActiveTab("analyze");
    } catch (_err) {
      setError("Upload failed. Try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const runComparison = async () => {
    if (!compareTextA.trim() || !compareTextB.trim()) {
      setCompareError("Please provide both transcripts to compare.");
      return;
    }

    setCompareLoading(true);
    setCompareError("");
    try {
      const compared = await compareTranscripts(compareTextA, compareTextB);
      setCompareResult(compared);
    } catch (_err) {
      setCompareResult(null);
      setCompareError("Comparison failed. Try again.");
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <h1 className="text-sm font-semibold tracking-wide text-slate-100">
            Financial Pragmatic AI (Stateless)
          </h1>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`rounded px-3 py-1 text-xs ${
                activeTab === "analyze" ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300"
              }`}
            >
              Analyze
            </button>
            <button
              onClick={() => setActiveTab("compare")}
              className={`rounded px-3 py-1 text-xs ${
                activeTab === "compare" ? "bg-sky-600 text-white" : "bg-slate-800 text-slate-300"
              }`}
            >
              Compare
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-4 px-4 py-4 lg:grid-cols-12">
        <aside className="space-y-4 lg:col-span-4">
          <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
            <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Transcript Input</h2>
            <textarea
              value={transcript}
              onChange={(event) => setTranscript(event.target.value)}
              rows={14}
              className="w-full rounded border border-slate-700 bg-slate-950 p-3 text-xs outline-none ring-0"
              disabled={loading}
            />
            <div className="mt-3 grid gap-2">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="rounded bg-sky-600 px-3 py-2 text-xs font-semibold text-white disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>

            <div className="mt-3 space-y-2 rounded border border-slate-800 bg-slate-950 p-3">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">Upload Transcript</p>
              <input
                type="file"
                accept=".txt,.pdf,text/plain,application/pdf"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
                disabled={loading}
                className="w-full text-xs text-slate-300"
              />
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="rounded bg-slate-700 px-3 py-2 text-xs font-semibold text-slate-100 disabled:opacity-50"
              >
                {loading ? "Uploading..." : "Upload & Analyze"}
              </button>
            </div>

            {error ? <p className="mt-3 text-xs text-rose-400">{error}</p> : null}
          </section>
        </aside>

        <section className="space-y-4 lg:col-span-8">
          {activeTab === "analyze" ? (
            <>
              <SummaryCard
                signal={result?.signal}
                score={result?.score}
                prediction={result?.prediction}
                confidence={result?.confidence}
                volatility={result?.volatility}
                keyDriver={drivers.growth[0]}
                keyConcern={drivers.risk[0]}
              />

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Timeline</h2>
                <div className="h-72 w-full">
                  <TimelineChart segments={segments} />
                </div>
              </section>

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Signal Heatmap</h2>
                <SignalHeatmap segments={segments} />
              </section>

              <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Signal Distribution</h2>
                <div className="space-y-2">
                  {signalDistribution.map((row) => (
                    <div key={row.intent} className="grid grid-cols-[150px_1fr_50px] items-center gap-2 text-xs">
                      <span className="text-slate-400">{row.intent}</span>
                      <div className="h-1.5 rounded bg-slate-800">
                        <div className="h-full rounded bg-sky-500" style={{ width: `${row.percentage}%` }} />
                      </div>
                      <span className="text-right text-slate-400">{row.percentage}%</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-emerald-500/40 bg-slate-900 p-4">
                  <h3 className="mb-2 text-xs uppercase tracking-wider text-emerald-300">Growth Drivers</h3>
                  <ul className="max-h-48 list-disc space-y-1 overflow-y-auto pl-4 text-xs text-slate-300">
                    {(drivers.growth.length ? drivers.growth : ["No growth driver detected"]).map(
                      (driver, index) => (
                        <li key={`g-${index}`} title={driver}>
                          {driver}
                        </li>
                      )
                    )}
                  </ul>
                </div>

                <div className="rounded-lg border border-rose-500/40 bg-slate-900 p-4">
                  <h3 className="mb-2 text-xs uppercase tracking-wider text-rose-300">Risk Drivers</h3>
                  <ul className="max-h-48 list-disc space-y-1 overflow-y-auto pl-4 text-xs text-slate-300">
                    {(drivers.risk.length ? drivers.risk : ["No risk concern detected"]).map(
                      (driver, index) => (
                        <li key={`r-${index}`} title={driver}>
                          {driver}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </section>
            </>
          ) : null}

          {activeTab === "compare" ? (
            <section className="space-y-4 rounded-lg border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-xs uppercase tracking-wider text-slate-400">Compare Transcripts</h2>
              <div className="grid gap-3 lg:grid-cols-2">
                <textarea
                  rows={10}
                  value={compareTextA}
                  onChange={(event) => setCompareTextA(event.target.value)}
                  className="w-full rounded border border-slate-700 bg-slate-950 p-3 text-xs outline-none ring-0"
                  placeholder="Transcript A"
                />
                <textarea
                  rows={10}
                  value={compareTextB}
                  onChange={(event) => setCompareTextB(event.target.value)}
                  className="w-full rounded border border-slate-700 bg-slate-950 p-3 text-xs outline-none ring-0"
                  placeholder="Transcript B"
                />
              </div>

              <button
                onClick={runComparison}
                disabled={compareLoading}
                className="rounded bg-sky-600 px-3 py-2 text-xs font-semibold text-white disabled:opacity-50"
              >
                {compareLoading ? "Comparing..." : "Compare"}
              </button>

              {compareError ? <p className="text-xs text-rose-400">{compareError}</p> : null}

              {compareResult ? (
                <div className="space-y-2 rounded border border-slate-700 bg-slate-950 p-3 text-xs">
                  <p>
                    Signal shift: {compareResult.signal_difference?.from} → {compareResult.signal_difference?.to}
                  </p>
                  <p>Risk delta: {compareResult.risk_delta}%</p>
                  <p>Confidence delta: {compareResult.confidence_delta}%</p>
                  <p>Trend: {compareResult.trend}</p>
                  <p className="text-slate-400">{compareResult.comparison}</p>
                </div>
              ) : null}
            </section>
          ) : null}
        </section>
      </main>
    </div>
  );
}
