import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
);

const intentWeights = {
  EXPANSION: 1,
  COST_PRESSURE: -1,
  STRATEGIC_PROBING: -0.5,
  GENERAL_UPDATE: 0,
};

const intentPointColors = {
  EXPANSION: "#00ff9c",
  COST_PRESSURE: "#ff4d4f",
  STRATEGIC_PROBING: "#facc15",
  GENERAL_UPDATE: "#9ca3af",
};

function smoothData(data, windowSize = 5) {
  return data.map((_, i, arr) => {
    const start = Math.max(0, i - windowSize);
    const end = Math.min(arr.length, i + windowSize);
    const subset = arr.slice(start, end);
    return subset.reduce((a, b) => a + b, 0) / subset.length;
  });
}

export default function TimelineChart({ segments }) {
  const labels = segments.map((_, i) => `Step ${i + 1}`);
  const signalData = segments.map((s) => intentWeights[s.intent] ?? 0);
  const smoothedData = smoothData(signalData);
  const pointColors = segments.map(
    (segment) => intentPointColors[segment.intent] || intentPointColors.GENERAL_UPDATE
  );

  const data = {
    labels,
    datasets: [
      {
        label: "Conversation Flow",
        data: smoothedData,
        fill: true,
        tension: 0.45,
        borderWidth: 2,
        borderColor: "#3b82f6",
        backgroundColor: (context) => {
          const chart = context.chart;
          const { ctx, chartArea } = chart;
          if (!chartArea) {
            return "rgba(59, 130, 246, 0.15)";
          }
          const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          gradient.addColorStop(0, "rgba(59, 130, 246, 0.30)");
          gradient.addColorStop(1, "rgba(59, 130, 246, 0.03)");
          return gradient;
        },
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: pointColors,
        pointBorderColor: "#0b0f14",
        pointBorderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#121821",
        titleColor: "#d1d5db",
        bodyColor: "#d1d5db",
        borderColor: "#1f2a37",
        borderWidth: 1,
        callbacks: {
          title: (items) => {
            const index = items[0]?.dataIndex ?? 0;
            return `Step ${index + 1}`;
          },
          label: (item) => {
            const index = item.dataIndex;
            const segment = segments[index];
            const intent = segment?.intent || "GENERAL_UPDATE";
            return `Intent: ${intent}`;
          },
          afterLabel: (item) => {
            const index = item.dataIndex;
            const segment = segments[index];
            const text = segment?.text || "";
            return `Text: ${text}`;
          },
        },
      },
    },
    animations: {
      tension: {
        duration: 700,
        easing: "easeOutCubic",
      },
    },
    scales: {
      x: {
        ticks: { color: "#9ca3af" },
        grid: { color: "#1f2a37" },
      },
      y: {
        ticks: {
          color: "#9ca3af",
          stepSize: 0.5,
          callback: (value) => {
            if (value === 1) return "Growth";
            if (value === -1) return "Risk";
            if (value === -0.5) return "Probe";
            return "Neutral";
          },
        },
        min: -1,
        max: 1,
        grid: { color: "#1f2a37" },
      },
    },
  };

  return <Line data={data} options={options} />;
}
