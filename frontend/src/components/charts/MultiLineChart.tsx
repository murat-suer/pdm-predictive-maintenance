import {
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { axisDefaults, tooltipDefaults } from './palette'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend)

export interface LineSeriesData {
  id: string
  label: string
  values: number[]
  color: string
}

interface MultiLineChartProps {
  labels: string[]
  series: LineSeriesData[]
  ariaLabel: string
  height?: number
  yMin?: number
  yMax?: number
  formatValue?: (v: number) => string
  /** Show the built-in chart.js legend */
  showLegend?: boolean
}

// ISA-101 colorblind-safe: each series gets a DISTINCT dash pattern so the
// line can be identified without relying on color alone.
const DASH_PATTERNS: number[][] = [
  [],           // solid
  [6, 3],       // dash
  [2, 2],       // dot
  [6, 3, 2, 3], // dash-dot
  [10, 4],      // long dash
  [1, 3],       // dot (tight)
  [8, 3, 2, 3, 2, 3], // dash-dot-dot
  [14, 4],      // extra long dash
]

export default function MultiLineChart({
  labels,
  series,
  ariaLabel,
  height = 220,
  yMin,
  yMax,
  formatValue,
  showLegend = false,
}: MultiLineChartProps) {
  if (series.length === 0 || labels.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-text-tertiary text-xs"
        style={{ height }}
        aria-label={ariaLabel}
      >
        —
      </div>
    )
  }

  const fmt = formatValue ?? ((v: number) => v.toFixed(1))

  const datasets = series.map((s, idx) => ({
    label: s.label,
    data: s.values,
    borderColor: s.color,
    backgroundColor: s.color,
    borderWidth: 1.5,
    borderDash: DASH_PATTERNS[idx % DASH_PATTERNS.length],
    tension: 0.25,
    pointRadius: 0,
    pointHoverRadius: 3,
  }))

  return (
    <div aria-label={ariaLabel}>
      <div style={{ height }}>
        <Line
          data={{ labels, datasets }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: {
                display: showLegend,
                labels: {
                  color: '#a0a0a8',
                  font: { family: "'IBM Plex Mono', monospace", size: 9 },
                  boxWidth: 10,
                  padding: 6,
                },
              },
              tooltip: {
                ...tooltipDefaults,
                callbacks: {
                  label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y != null ? fmt(ctx.parsed.y) : '—'}`,
                },
              },
            },
            scales: {
              x: {
                ...axisDefaults,
                ticks: { ...axisDefaults.ticks, maxTicksLimit: 8, maxRotation: 0 },
              },
              y: {
                ...axisDefaults,
                min: yMin,
                max: yMax,
                ticks: {
                  ...axisDefaults.ticks,
                  maxTicksLimit: 5,
                  callback: (v) => (v == null ? '' : fmt(Number(v))),
                },
              },
            },
          }}
        />
      </div>
      {/* Custom HTML legend — shows color swatch + dash pattern sample + label.
          This satisfies ISA-101: identity is NOT conveyed by color alone. */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {series.map((s, idx) => {
          const dash = DASH_PATTERNS[idx % DASH_PATTERNS.length]
          // Build an SVG line-style sample that mirrors the actual borderDash
          const svgW = 24
          const svgH = 8
          const strokeDasharray = dash.length > 0 ? dash.join(',') : undefined

          return (
            <span key={s.id} className="flex items-center gap-1.5 text-[10px] text-text-secondary">
              {/* SVG renders the actual dash pattern, not just a color block */}
              <svg
                width={svgW}
                height={svgH}
                viewBox={`0 0 ${svgW} ${svgH}`}
                aria-hidden="true"
                style={{ flexShrink: 0 }}
              >
                <line
                  x1="0"
                  y1={svgH / 2}
                  x2={svgW}
                  y2={svgH / 2}
                  stroke={s.color}
                  strokeWidth="2"
                  strokeDasharray={strokeDasharray}
                />
              </svg>
              {s.label}
            </span>
          )
        })}
      </div>
    </div>
  )
}
