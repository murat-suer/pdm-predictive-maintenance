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

  const datasets = series.map((s) => ({
    label: s.label,
    data: s.values,
    borderColor: s.color,
    backgroundColor: s.color,
    borderWidth: 1.5,
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
      {/* Custom HTML legend — one row per machine, space-efficient */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
        {series.map((s) => (
          <span key={s.id} className="flex items-center gap-1 text-[10px] text-text-secondary">
            <span
              className="inline-block w-3 h-0.5 rounded"
              style={{ backgroundColor: s.color }}
            />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  )
}
