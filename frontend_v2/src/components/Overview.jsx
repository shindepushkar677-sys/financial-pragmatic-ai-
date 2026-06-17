import { motion } from "framer-motion";
import TimelineChart from "./TimelineChart";

export default function Overview({ result }) {
  if (!result) return null;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="transition-all duration-300">
      <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg mb-4 transition-all duration-300">
        <h2 className="mb-3 text-lg">Distribution</h2>
        {[
          { key: "growth", label: "Growth", color: "bg-green-500" },
          { key: "risk", label: "Risk", color: "bg-red-500" },
          { key: "neutral", label: "Neutral", color: "bg-yellow-500" },
        ].map((item) => {
          const value = result?.distribution?.[item.key] || 0;
          const formattedPercent = Math.round(value * 100) + "%";
          return (
            <div key={item.key} className="mb-3 last:mb-0">
              <div className="flex justify-between text-sm mb-1">
                <span>{item.label}</span>
                <span>{formattedPercent}</span>
              </div>
              <div className="w-full h-2 bg-[#0d1117] rounded">
                <div
                  className={`h-2 rounded ${item.color}`}
                  style={{ width: formattedPercent }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <TimelineChart timeline={result?.timeline} />

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-300">
          <h3 className="text-green-400">Growth Drivers</h3>
          <ul className="text-sm mt-2 space-y-1">
            {result?.drivers?.growth?.length ? (
              result.drivers.growth.map((item, idx) => <li key={idx}>{item}</li>)
            ) : (
              <li>No growth drivers</li>
            )}
          </ul>
        </div>

        <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg transition-all duration-300">
          <h3 className="text-red-400">Risk Drivers</h3>
          <ul className="text-sm mt-2 space-y-1">
            {result?.drivers?.risk?.length ? (
              result.drivers.risk.map((item, idx) => <li key={idx}>{item}</li>)
            ) : (
              <li>No risk drivers</li>
            )}
          </ul>
        </div>
      </div>
    </motion.div>
  );
}
