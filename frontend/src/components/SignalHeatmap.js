const colorMap = {
  EXPANSION: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300",
  COST_PRESSURE: "bg-rose-500/15 border-rose-500/40 text-rose-300",
  STRATEGIC_PROBING: "bg-amber-500/15 border-amber-500/40 text-amber-300",
  GENERAL_UPDATE: "bg-slate-500/15 border-slate-500/40 text-slate-300",
};

export default function SignalHeatmap({ segments }) {
  const counts = {};
  segments.forEach((segment) => {
    counts[segment.intent] = (counts[segment.intent] || 0) + 1;
  });

  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
      {Object.entries(counts).map(([intent, count]) => (
        <div
          key={intent}
          className={`rounded border p-3 text-center transition hover:scale-[1.02] ${
            colorMap[intent] || "bg-slate-800/40 border-slate-600 text-slate-300"
          }`}
          title={intent}
        >
          <p className="text-[10px] uppercase tracking-wide">{intent}</p>
          <p className="text-xl font-bold">{count}</p>
        </div>
      ))}
    </div>
  );
}
