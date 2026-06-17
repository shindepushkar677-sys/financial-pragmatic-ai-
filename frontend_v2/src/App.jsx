import { useEffect, useMemo, useRef, useState } from "react";
import { supabase } from "./supabaseClient";
import Auth from "./components/Auth";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Tabs from "./components/Tabs";

const API_URL = "http://localhost:8000/analyze";

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [session, setSession] = useState(null);
  const [history, setHistory] = useState([]);
  const [compareA, setCompareA] = useState(null);
  const [compareB, setCompareB] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [isFromHistory, setIsFromHistory] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (compareA && compareB) {
      setActiveTab("compare");
    }
  }, [compareA, compareB]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session) fetchHistory(session.user.id);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      if (session) fetchHistory(session.user.id);
      else setHistory([]);
    });

    return () => subscription.unsubscribe();
  }, []);

  const fetchHistory = async (userId) => {
    const { data, error } = await supabase
      .from("analyses")
      .select("*")
      .eq("user_id", userId)
      .order("created_at", { ascending: false });

    if (!error && data) setHistory(data);
  };

  const handleHistoryClick = (item) => {
    setSelectedAnalysis(item);
    setIsFromHistory(true);
    setTranscript(item.transcript);
    setResult({
      signal: item.signal,
      score: item.score,
      distribution: item.distribution,
      drivers: {
        growth: item.growth_drivers,
        risk: item.risk_drivers,
      },
      timeline: item.timeline || [], // Fallback for old records
    });
  };


  const signal = (result?.signal || "neutral").toLowerCase();

  const scoreText =
    typeof result?.score === "number" ? result.score.toFixed(2) : "--";
  const confidenceText =
    typeof result?.confidence === "number"
      ? `${(result.confidence <= 1 ? result.confidence * 100 : result.confidence).toFixed(1)}%`
      : "85.0%";

  const barWidth = (value) => {
    const pct = value <= 1 ? value * 100 : value;
    return `${Math.max(0, Math.min(100, pct))}%`;
  };

  const heroSignalClass =
    signal === "growth"
      ? "text-[#00ff9c]"
      : signal === "risk"
        ? "text-[#ff4d4f]"
        : "text-[#58a6ff]";

  const heroGlowClass =
    signal === "growth"
      ? "shadow-[0_0_20px_rgba(0,255,156,0.2)]"
      : signal === "risk"
        ? "shadow-[0_0_20px_rgba(255,77,79,0.2)]"
        : "shadow-[0_0_20px_rgba(88,166,255,0.2)]";

  const handleAnalyze = async () => {
    if (!transcript.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: transcript }),
      });

      const data = await response.json();
      console.log("RAW BACKEND:", data);
      const intentDist = data.intent_distribution || {};
      const total = Object.values(intentDist).reduce((a, b) => a + b, 0) || 1;

      const mapped = {
        signal: data.final_signal || data.signal,
        score: data.score,
        confidence: data.confidence || 0.8,
        distribution: {
          growth: (intentDist.EXPANSION || 0) / total,
          risk: (intentDist.COST_PRESSURE || 0) / total,
          neutral: (intentDist.GENERAL_UPDATE || 0) / total,
        },
        drivers: {
          growth: data.growth_drivers || [],
          risk: data.risk_drivers || [],
        },
        timeline: data.timeline || [],
      };

      console.log("MAPPED:", mapped);
      setResult(mapped);

      // Save to Supabase
      if (session?.user) {
        if (isFromHistory && selectedAnalysis?.id) {
          const { data, error } = await supabase
            .from("analyses")
            .update({
              signal: mapped.signal,
              score: mapped.score,
              distribution: mapped.distribution,
              growth_drivers: mapped.drivers.growth,
              risk_drivers: mapped.drivers.risk,
              timeline: mapped.timeline,
            })
            .eq("id", selectedAnalysis.id);

          console.log("UPDATE DATA:", data);
          console.log("UPDATE ERROR:", error);
        } else {
          const { data, error } = await supabase.from("analyses").insert({
            user_id: session.user.id,
            transcript: transcript,
            signal: mapped.signal,
            score: mapped.score,
            distribution: mapped.distribution,
            growth_drivers: mapped.drivers.growth,
            risk_drivers: mapped.drivers.risk,
            timeline: mapped.timeline,
          });

          console.log("INSERT DATA:", data);
          console.log("INSERT ERROR:", error);

          setIsFromHistory(false);
          setSelectedAnalysis(null);
        }

        fetchHistory(session.user.id);
      }
    } catch (error) {
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const content = typeof reader.result === "string" ? reader.result : "";
      setTranscript(content);
      setUploadedFileName(file.name);
      setIsFromHistory(false);
      setSelectedAnalysis(null);
    };
    reader.readAsText(file);
  };

  if (!session) return <Auth />;

  return (
    <div className="flex h-screen overflow-hidden bg-[#0d1117] text-[#c9d1d9]">
      <Sidebar 
        history={history} 
        onHistoryClick={handleHistoryClick} 
        userEmail={session.user.email}
        compareMode={compareMode}
        setCompareMode={setCompareMode}
        compareA={compareA}
        setCompareA={setCompareA}
        compareB={compareB}
        setCompareB={setCompareB}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar userEmail={session.user.email} />

        <div className="p-4 flex-1 overflow-y-auto">
          <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg mb-4 transition-all duration-300">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              className="hidden"
              onChange={handleFileChange}
            />
            <textarea
              className="w-full h-32 bg-[#0d1117] border border-[#30363d] focus:border-blue-500 p-2 rounded transition-all duration-300 outline-none resize-none"
              value={transcript}
              onChange={(e) => {
                setTranscript(e.target.value);
                setIsFromHistory(false);
                setSelectedAnalysis(null);
              }}
              placeholder="Enter financial transcript..."
            />
            <div className="mt-2 flex items-center gap-2">
              <button
                className="bg-gradient-to-r from-blue-500 to-blue-700 hover:scale-105 active:scale-95 transition-all duration-300 px-6 py-2 rounded font-semibold disabled:opacity-60"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
              <button
                className="bg-[#30363d] px-4 py-2 rounded hover:bg-[#444c56] transition-all duration-300 text-sm"
                onClick={handleUploadClick}
                type="button"
              >
                Upload File
              </button>
            </div>
            {uploadedFileName ? (
              <div className="mt-2 text-xs text-[#8b949e]">
                Uploaded: {uploadedFileName}
              </div>
            ) : null}
          </div>

          <div className={`bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-6 rounded-lg mb-4 text-center transition-all duration-300 ${heroGlowClass}`}>
            <div className={`text-3xl font-bold transition-all duration-300 ${(result?.signal || "NEUTRAL").toUpperCase() === 'GROWTH' ? 'text-green-400' : (result?.signal || "NEUTRAL").toUpperCase() === 'RISK' ? 'text-red-400' : 'text-blue-400'}`}>
              {(result?.signal || "NEUTRAL").toUpperCase()}
            </div>
            <div className="text-xl mt-1">{scoreText}</div>
            <div className="text-xs text-[#8b949e] mt-2 uppercase tracking-widest">
              Confidence: <span className="text-[#c9d1d9]">{confidenceText}</span>
            </div>
          </div>

          <div className="flex justify-between items-center mb-3 mt-4">
            <div className="text-sm font-medium text-[#8b949e] italic transition-opacity tracking-wide pl-2">
              {compareMode && !compareA && !compareB && "Select 2 analyses to compare from sidebar"}
              {compareMode && ((compareA && !compareB) || (!compareA && compareB)) && "Select one more analysis"}
            </div>
            <button 
              onClick={() => setCompareMode(!compareMode)}
              className={`px-5 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider transition-all duration-300 ${
                compareMode 
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/50 shadow-[0_0_12px_rgba(59,130,246,0.3)] scale-[1.02]" 
                  : "bg-[#30363d] text-[#8b949e] hover:bg-[#444c56] border border-transparent"
              }`}
            >
              Compare Mode
            </button>
          </div>

          <Tabs 
            active={activeTab}
            onTabChange={setActiveTab}
            result={result} 
            compareA={compareA}
            compareB={compareB}
            onClearCompare={() => {
              setCompareA(null);
              setCompareB(null);
            }}
          />
        </div>
      </div>
    </div>
  );
}
