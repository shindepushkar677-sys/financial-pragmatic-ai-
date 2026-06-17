import { User } from "lucide-react";

export default function Navbar({ userEmail }) {
  return (
    <div className="h-14 flex items-center justify-between px-6 border-b border-[#30363d] bg-[rgba(22,27,34,0.6)] backdrop-blur-xl transition-all duration-300">
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
        <h1 className="font-bold text-sm tracking-wider uppercase text-[#c9d1d9]">
          Financial <span className="text-blue-400">Pragmatic</span> AI
        </h1>
      </div>

      {userEmail && (
        <div className="flex items-center gap-2 text-[#8b949e]">
          <span className="text-[10px] italic border-r border-[#30363d] pr-3 mr-1 uppercase tracking-tighter">Terminal Session</span>
          <User size={14} className="text-blue-500/70" />
          <span className="text-[11px] font-mono">{userEmail}</span>
        </div>
      )}
    </div>
  );
}
