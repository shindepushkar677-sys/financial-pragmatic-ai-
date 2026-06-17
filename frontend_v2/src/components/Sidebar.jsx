import { Home, BarChart2, Settings, History, LogOut, Clock, ToggleLeft, ToggleRight } from "lucide-react";
import { supabase } from "../supabaseClient";

export default function Sidebar({ history, onHistoryClick, userEmail, compareMode, setCompareMode, compareA, setCompareA, compareB, setCompareB }) {
  const handleLogout = async () => {
    await supabase.auth.signOut();
  };

  const handleItemClick = (item) => {
    if (!compareMode) {
      onHistoryClick(item);
    } else {
      if (!compareA) {
        setCompareA(item);
      } else if (!compareB && compareA.id !== item.id) {
        setCompareB(item);
      } else {
        setCompareA(compareB);
        setCompareB(item);
      }
    }
  };

  const getSignalColor = (signal) => {
    const s = signal?.toLowerCase();
    if (s === "growth" || s === "expansion") return "text-green-400 border-green-500/30 bg-green-500/10";
    if (s === "risk" || s === "cost_pressure") return "text-red-400 border-red-500/30 bg-red-500/10";
    return "text-blue-400 border-blue-500/30 bg-blue-500/10";
  };

  return (
    <div className="w-64 bg-[#0d1117] border-r border-[#30363d] flex flex-col h-full overflow-hidden transition-all duration-300">
      {/* Top Icons */}
      <div className="p-4 border-b border-[#30363d] flex justify-around">
        <Home className="text-[#8b949e] hover:text-white cursor-pointer transition-colors" size={20} title="Home" />
        <BarChart2 className="text-[#8b949e] hover:text-white cursor-pointer transition-colors" size={20} title="Analytics" />
        <Settings className="text-[#8b949e] hover:text-white cursor-pointer transition-colors" size={20} title="Settings" />
      </div>

      {/* History Section */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="p-4 flex items-center gap-2 text-[#8b949e] uppercase tracking-widest text-[10px] font-bold">
          <History size={14} />
          <span>Recent Analyses</span>
        </div>

        <div className="flex-1 overflow-y-auto px-2 space-y-1 custom-scrollbar">
          {history.length === 0 ? (
            <div className="p-4 text-center text-[#8b949e] text-xs italic">
              No past analyses found.
            </div>
          ) : (
            history.map((item) => {
              const isA = item.id === compareA?.id;
              const isB = item.id === compareB?.id;
              const isSelected = isA || isB;
              
              return (
                <div
                  key={item.id}
                  onClick={() => handleItemClick(item)}
                  className={`p-3 rounded-md border cursor-pointer transition-all duration-200 group ${
                    isSelected
                      ? 'border-blue-500/50 bg-[#1e2532] scale-[1.02] shadow-sm z-10 relative' 
                      : 'border-transparent hover:border-[#30363d] hover:bg-[#1c212a]'
                  }`}
                >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold ${getSignalColor(item.signal)}`}>
                      {item.signal?.replace('_', ' ')}
                    </span>
                    {isA && <span className="bg-blue-600 text-white text-[9px] px-2 py-0.5 rounded-full font-bold">A</span>}
                    {isB && <span className="bg-purple-600 text-white text-[9px] px-2 py-0.5 rounded-full font-bold">B</span>}
                  </div>
                  <span className="text-[9px] text-[#8b949e] flex items-center gap-1">
                    <Clock size={10} />
                    {new Date(item.created_at).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                  </span>
                </div>
                <p className="text-[11px] text-[#8b949e] line-clamp-2 leading-relaxed group-hover:text-[#c9d1d9] transition-colors">
                  {item.transcript}
                </p>
              </div>
              )
            })
          )}
        </div>
      </div>

      {/* Footer / User */}
      <div className="p-4 border-t border-[#30363d] bg-[#161b22]/50">
        <div className="flex items-center justify-between">
          <div className="flex flex-col min-w-0">
            <span className="text-[10px] text-[#8b949e] truncate">Logged in as</span>
            <span className="text-xs font-medium text-white truncate" title={userEmail}>
              {userEmail?.split('@')[0]}
            </span>
          </div>
          <button
            onClick={handleLogout}
            className="p-2 text-[#8b949e] hover:text-red-400 hover:bg-red-400/10 rounded-md transition-all"
            title="Sign Out"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #444c56; }
      `}} />
    </div>
  );
}
