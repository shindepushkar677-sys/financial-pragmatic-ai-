import React from 'react';

export default function Compare({ compareA, compareB, onClearCompare }) {
  if (!compareA || !compareB) {
    return (
      <div className="text-[#8b949e] text-center p-8 bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] rounded-lg">
        Select 2 analyses from the sidebar to compare
      </div>
    );
  }

  const delta = (compareB.score || 0) - (compareA.score || 0);
  const deltaColor = delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-[#8b949e]";
  const deltaSign = delta > 0 ? "+" : "";

  const getInterpretation = (d) => {
    if (d > 0.05) return "Strong Improvement";
    if (d > 0.01) return "Slight Improvement";
    if (d > -0.01) return "No Change";
    if (d > -0.05) return "Slight Decline";
    return "Strong Decline";
  };

  const getSummary = () => {
    const dGrowth = (compareB.distribution?.growth || 0) - (compareA.distribution?.growth || 0);
    const dRisk = (compareB.distribution?.risk || 0) - (compareA.distribution?.risk || 0);

    if (dGrowth > dRisk && dGrowth > 0.01) return "Growth sentiment improved slightly, driven by stronger revenue signals.";
    if (dRisk > dGrowth && dRisk > 0.01) return "Risk sentiment increased, highlighting emerging cost or operational pressures.";
    return "Overall intent sentiment remains relatively mixed and stable across both periods.";
  };

  const cleanDrivers = (drivers) => {
    if (!drivers) return [];
    return drivers.filter(d => /\d/.test(d) || d.split(" ").filter(w => w.trim().length > 0).length >= 3);
  };

  const UnifiedDrivers = ({ title, color, titleColor, fieldA, fieldB }) => {
    const a = fieldA || [];
    const b = fieldB || [];
    
    const countA = a.reduce((acc, curr) => ({ ...acc, [curr]: (acc[curr] || 0) + 1 }), {});
    const countB = b.reduce((acc, curr) => ({ ...acc, [curr]: (acc[curr] || 0) + 1 }), {});

    const newDrivers = [];
    Object.keys(countB).forEach(d => {
      const diff = countB[d] - (countA[d] || 0);
      for (let i = 0; i < diff; i++) newDrivers.push(d);
    });

    const removedDrivers = [];
    Object.keys(countA).forEach(d => {
      const diff = countA[d] - (countB[d] || 0);
      for (let i = 0; i < diff; i++) removedDrivers.push(d);
    });

    const keptDrivers = [];
    Object.keys(countA).forEach(d => {
      const min = Math.min(countA[d], countB[d] || 0);
      for (let i = 0; i < min; i++) keptDrivers.push(d);
    });

    return (
      <div className={`bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-200 hover:-translate-y-[2px]`}>
        <h3 className={`${titleColor} font-semibold mb-4 border-b border-[#30363d] pb-2`}>{title}</h3>
        <ul className="space-y-2 text-sm text-[#8b949e]">
          {newDrivers.map((d, i) => (
            <li key={`new-${i}`} className="flex gap-2 items-start text-[#c9d1d9]" title={d}>
              <span className="text-[9px] font-bold text-green-400 bg-green-400/10 px-1 py-0.5 rounded border border-green-400/20 mt-0.5 min-w-[40px] text-center tracking-widest uppercase">NEW</span> 
              <span className="pt-0.5 leading-snug">{d}</span>
            </li>
          ))}
          {removedDrivers.map((d, i) => (
            <li key={`rem-${i}`} className="flex gap-2 items-start opacity-75" title={d}>
              <span className="text-[9px] font-bold text-red-400 bg-red-400/10 px-1 py-0.5 rounded border border-red-400/20 mt-0.5 min-w-[40px] text-center tracking-widest uppercase">REMOVED</span> 
              <span className="pt-0.5 leading-snug line-through text-[#c9d1d9]">{d}</span>
            </li>
          ))}
          {keptDrivers.map((d, i) => (
            <li key={`kept-${i}`} className="flex gap-2 items-start" title={d}>
              <span className="text-[9px] font-bold text-[#8b949e] px-1 py-0.5 min-w-[40px] text-center opacity-0 mt-0.5">---</span> 
              <span className="pt-0.5 leading-snug">{d}</span>
            </li>
          ))}
          {!newDrivers.length && !removedDrivers.length && !keptDrivers.length && (
            <li className="italic text-center py-2 opacity-50">No drivers detected</li>
          )}
        </ul>
      </div>
    );
  };

  const DistRow = ({ label, keyName }) => {
    const valA = Math.round((compareA.distribution?.[keyName] || 0) * 100);
    const valB = Math.round((compareB.distribution?.[keyName] || 0) * 100);
    const diff = valB - valA;
    const diffColor = diff > 0 ? "text-green-400" : diff < 0 ? "text-red-400" : "text-[#8b949e]";
    const sign = diff > 0 ? "+" : "";

    return (
      <div className="space-y-1.5 py-2 border-b border-[#30363d]/30 last:border-0">
        <div className="flex justify-between items-center text-sm font-medium">
          <span className="capitalize w-16 text-[#c9d1d9]">{label}:</span>
          <span className="text-[#8b949e] w-20 text-right">{valA}% <span className="mx-1">→</span> <span className="text-white">{valB}%</span></span>
          <span className={`w-14 text-right ${diffColor}`}>({sign}{diff}%)</span>
        </div>
        <div className="h-1.5 bg-[#0d1117] rounded-full overflow-hidden flex relative w-full">
          <div className="absolute top-0 bottom-0 left-0 bg-blue-500/30 transition-all duration-500 ease-in-out" style={{ width: `${valA}%` }} />
          <div className="absolute top-0 bottom-0 left-0 bg-blue-500/80 transition-all duration-500 ease-in-out" style={{ width: `${valB}%` }} />
        </div>
      </div>
    );
  };

  const getTimelineInsight = () => {
    if (!compareB.timeline || compareB.timeline.length < 2) return null;
    let changes = 0;
    for (let i = 1; i < compareB.timeline.length; i++) {
      if (compareB.timeline[i].value !== compareB.timeline[i-1].value) changes++;
    }
    return changes > compareB.timeline.length * 0.4 ? "Volatility Increased" : "Trend Stability: Improved";
  };
  const timelineInsight = getTimelineInsight();

  const formatDate = (dateString) => {
    if (!dateString) return null;
    return new Date(dateString).toLocaleDateString("en-GB", { day: 'numeric', month: 'short' });
  };
  const dateA = formatDate(compareA.created_at);
  const dateB = formatDate(compareB.created_at);

  return (
    <div className="space-y-8">
      {/* Header & Summary */}
      <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-5 rounded-lg transition-all duration-200 hover:-translate-y-[2px]">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-lg font-bold text-white">Comparison Analysis</h2>
            <div className="text-xs text-[#8b949e] mt-0.5 uppercase tracking-wider">A (Older) <span className="mx-1">vs</span> B (Newer)</div>
            {dateA && dateB && (
              <div className="text-[10px] text-[#c9d1d9] mt-1 tracking-wider uppercase font-medium">{dateA} <span className="mx-1 text-[#8b949e]">→</span> {dateB}</div>
            )}
          </div>
          <button 
            onClick={onClearCompare}
            className="bg-[#30363d]/80 hover:bg-[#444c56] hover:text-white px-4 py-2 rounded text-xs font-semibold uppercase tracking-wider transition-all duration-300 hover:brightness-110 hover:shadow-[0_0_12px_rgba(255,255,255,0.1)]"
          >
            Clear Comparison
          </button>
        </div>
        <div className="text-sm text-blue-300 italic border-l-2 border-blue-500/50 pl-3">
          {getSummary()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Signal & Score */}
        <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-200 hover:-translate-y-[2px] flex flex-col">
          <h3 className="text-[#c9d1d9] font-semibold mb-4 border-b border-[#30363d] pb-2">Signal Shift</h3>
          <div className="flex-1 flex flex-col justify-center">
            <div className="flex items-center justify-between text-lg mb-4">
              <span className="uppercase font-bold text-white tracking-wide">{compareA.signal?.replace('_', ' ')}</span>
              <span className="text-[#8b949e] font-light">→</span>
              <span className="uppercase font-bold text-white tracking-wide">{compareB.signal?.replace('_', ' ')}</span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-[#8b949e] font-serif pr-2">Score: {compareA.score?.toFixed(2)}</span>
              <div className="flex flex-col items-center flex-1">
                <span className={`text-2xl font-bold tracking-tight ${deltaColor}`}>{deltaSign}{delta.toFixed(2)}</span>
              </div>
              <span className="text-sm text-[#8b949e] font-serif pl-2">Score: {compareB.score?.toFixed(2)}</span>
            </div>
            <div className="text-center w-full mt-1">
              <div className={`text-[11px] font-medium tracking-widest uppercase ${deltaColor}`}>{getInterpretation(delta)}</div>
            </div>
          </div>
        </div>

        {/* Intent Distribution */}
        <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-200 hover:-translate-y-[2px]">
          <h3 className="text-[#c9d1d9] font-semibold mb-2 border-b border-[#30363d] pb-2">Distribution Summary</h3>
          <div className="flex flex-col justify-center h-[calc(100%-40px)]">
            <DistRow label="growth" keyName="growth" />
            <DistRow label="neutral" keyName="neutral" />
            <DistRow label="risk" keyName="risk" />
          </div>
        </div>
      </div>

      {/* Drivers */}
      <div className="grid grid-cols-2 gap-4">
        <UnifiedDrivers 
          title="Growth Drivers Transition" 
          titleColor="text-green-400"
          fieldA={cleanDrivers(compareA.growth_drivers)} 
          fieldB={cleanDrivers(compareB.growth_drivers)} 
        />
        <UnifiedDrivers 
          title="Risk Drivers Transition" 
          titleColor="text-red-400"
          fieldA={cleanDrivers(compareA.risk_drivers)} 
          fieldB={cleanDrivers(compareB.risk_drivers)} 
        />
      </div>

      {/* Timeline Insight */}
      {timelineInsight && (
        <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-200 hover:-translate-y-[2px] text-center border-t-2 border-t-blue-500/20">
          <span className="text-sm text-[#8b949e] uppercase tracking-wide mr-2">Timeline Insight:</span>
          <span className="text-sm font-semibold text-white tracking-wide">{timelineInsight}</span>
        </div>
      )}
    </div>
  );
}
