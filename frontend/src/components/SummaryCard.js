function toneClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (["growth", "up", "low"].includes(normalized)) return "text-emerald-400";
  if (["risk", "down", "high"].includes(normalized)) return "text-rose-400";
  return "text-amber-300";
}

export default function SummaryCard({
  signal,
  score,
  prediction,
  confidence,
  volatility,
  keyDriver,
  keyConcern,
}) {
  const scoreTone = score > 65 ? "risk" : score < 35 ? "growth" : "neutral";

  return (
    <section className="rounded-lg border border-slate-700 bg-slate-900 p-4">
      <h2 className="mb-3 text-xs uppercase tracking-wider text-slate-400">Summary</h2>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <Metric label="Signal" value={String(signal || "neutral").toUpperCase()} tone={signal} />
        <Metric label="Risk Score" value={score ?? "-"} tone={scoreTone} />
        <Metric
          label="Prediction"
          value={String(prediction || "NEUTRAL").toUpperCase()}
          tone={prediction}
        />
        <Metric label="Confidence" value={`${confidence ?? "-"}%`} />
        <Metric
          label="Volatility"
          value={String(volatility || "-").toUpperCase()}
          tone={volatility}
        />
      </div>

      <div className="mt-3 space-y-1 text-xs leading-relaxed text-slate-300">
        <p>
          <span className="text-slate-500">Key driver:</span> {keyDriver || "No growth driver detected"}
        </p>
        <p>
          <span className="text-slate-500">Key concern:</span> {keyConcern || "No risk concern detected"}
        </p>
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded bg-slate-800">
        <div
          className="h-full bg-sky-500 transition-all"
          style={{ width: `${Math.max(0, Math.min(100, Number(confidence) || 0))}%` }}
        />
      </div>
      <p className="mt-1 text-[11px] text-slate-500">Confidence</p>
    </section>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className="rounded border border-slate-700 bg-slate-950 p-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`text-sm font-semibold ${toneClass(tone)}`}>{value}</p>
    </div>
  );
}
