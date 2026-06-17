import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

const INTENT_COLOR = {
  EXPANSION: "#00ff9c",
  COST_PRESSURE: "#ff4d4f",
  GENERAL_UPDATE: "#facc15",
  STRATEGIC_PROBING: "#facc15",
};

const INTENT_LABEL = {
  EXPANSION: "Growth",
  COST_PRESSURE: "Risk",
  GENERAL_UPDATE: "Neutral",
  STRATEGIC_PROBING: "Neutral",
};

// Subtle custom dot
function SubtleDot(props) {
  const { cx, cy, payload } = props;
  const color = INTENT_COLOR[payload?.intent] || "#8b949e";
  return (
    <circle
      cx={cx}
      cy={cy}
      r={2}
      fill={color}
      opacity={0.5}
    />
  );
}

function ActiveDot(props) {
  const { cx, cy, payload } = props;
  const color = INTENT_COLOR[payload?.intent] || "#8b949e";
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={color}
      style={{ filter: `drop-shadow(0 0 4px ${color})` }}
    />
  );
}

// Custom tooltip
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const color = INTENT_COLOR[d.intent] || "#8b949e";
  return (
    <div
      style={{
        background: "rgba(13,17,23,0.92)",
        border: `1px solid ${color}`,
        borderRadius: 8,
        padding: "10px 14px",
        maxWidth: 260,
        boxShadow: `0 0 12px ${color}44`,
      }}
    >
      <div style={{ color, fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
        Step {d.step} — {INTENT_LABEL[d.intent]}
      </div>
      <div style={{ color: "#8b949e", fontSize: 11, lineHeight: 1.5 }}>
        {d.label}
      </div>
    </div>
  );
}

// Y-axis zone labels
function ZoneLabel({ viewBox, label, color }) {
  return (
    <text
      x={viewBox.x + 6}
      y={viewBox.y + 13}
      fill={color}
      fontSize={10}
      opacity={0.35}
      fontWeight={600}
    >
      {label}
    </text>
  );
}

export default function TimelineChart({ timeline: rawTimeline }) {
  if (!rawTimeline || rawTimeline.length === 0) return null;

  // Downsample data
  const stepSize = Math.max(1, Math.floor(rawTimeline.length / 100));
  const timeline = rawTimeline.filter((_, i) => i % stepSize === 0);

  // Compute trend summary
  const avg = timeline.length > 0 ? timeline.reduce((a, b) => a + b.value, 0) / timeline.length : 0;
  
  let trendText = "Trend: Neutral/Mixed";
  let trendColor = "#facc15";
  if (avg > 0.2) {
    trendText = "Trend: Positive";
    trendColor = "#00ff9c";
  } else if (avg < -0.2) {
    trendText = "Trend: Risk-leaning";
    trendColor = "#ff4d4f";
  }

  return (
    <div className="bg-[rgba(22,27,34,0.6)] backdrop-blur-xl border border-[#30363d] p-4 rounded-lg mb-4 transition-all duration-300">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="mb-1 text-lg">Event Timeline</h2>
          <p className="text-xs text-[#8b949e]">
            Signal trajectory across transcript segments
          </p>
        </div>
        <div style={{ color: trendColor }} className="text-sm font-bold border border-[#30363d] px-3 py-1 rounded bg-[rgba(13,17,23,0.5)]">
          {trendText}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart
          data={timeline}
          margin={{ top: 10, right: 16, left: -20, bottom: 4 }}
        >
          {/* Background grid */}
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#21262d"
            vertical={false}
          />

          {/* Zone reference lines - reduced opacity */}
          <ReferenceLine
            y={0.5}
            stroke="#00ff9c"
            strokeOpacity={0.04}
            strokeWidth={28}
            label={<ZoneLabel label="GROWTH ZONE" color="#00ff9c" />}
          />
          <ReferenceLine
            y={-0.5}
            stroke="#ff4d4f"
            strokeOpacity={0.04}
            strokeWidth={28}
            label={<ZoneLabel label="RISK ZONE" color="#ff4d4f" />}
          />
          <ReferenceLine y={0} stroke="#30363d" strokeDasharray="4 4" strokeOpacity={0.5} />

          <XAxis
            dataKey="step"
            tick={{ fill: "#8b949e", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }}
            tickLine={false}
            label={{
              value: "Segment",
              position: "insideBottomRight",
              offset: -4,
              fill: "#8b949e",
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[-1.3, 1.3]}
            ticks={[-1, 0, 1]}
            tickFormatter={(v) => (v === 1 ? "+1" : v === -1 ? "−1" : "0")}
            tick={{ fill: "#8b949e", fontSize: 11 }}
            axisLine={{ stroke: "#30363d" }}
            tickLine={false}
          />

          <Tooltip content={<CustomTooltip />} />

          <Area
            type="monotone"
            dataKey="value"
            stroke="#58a6ff"
            strokeWidth={3}
            fill="rgba(88,166,255,0.12)"
            dot={<SubtleDot />}
            activeDot={<ActiveDot />}
            isAnimationActive={true}
            animationDuration={800}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
