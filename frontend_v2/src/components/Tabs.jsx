import { useState } from "react";
import Overview from "./Overview";
import Insights from "./Insights";
import Compare from "./Compare";

export default function Tabs({ active, onTabChange, result, compareA, compareB, onClearCompare }) {
return ( <div className="transition-all duration-300"> <div className="flex border-b border-[#30363d] transition-all duration-300">
{["overview", "insights", "compare"].map((tab) => (
<button
key={tab}
onClick={() => onTabChange(tab)}
className={`px-4 py-2 text-sm transition-all duration-300 ${
              active === tab
                ? "text-white border-b-2 border-blue-500"
                : "text-[#8b949e]"
            }`}
>
{tab.toUpperCase()} </button>
))} </div>

  <div className="p-4 transition-all duration-300">
    {active === "overview" && <Overview result={result} />}
    {active === "insights" && <Insights result={result} />}
    {active === "compare" && <Compare compareA={compareA} compareB={compareB} onClearCompare={onClearCompare} />}
  </div>
</div>
);
}
