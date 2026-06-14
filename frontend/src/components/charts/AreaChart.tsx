import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { axisDefaults, tooltipDefaults } from './palette'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip)

interface AreaChartProps {
  labels: string[]
  values: number[]
  color: string
  ariaLabel: string
  height?: number
  yLabel?: string
  /** Format function for tooltip values */
  formatValue?: (v: number) => string
}

export default function AreaChart({
  labels,
  values,
  color,
  ariaLabel,
  height = 180,
  formatValue,
}: AreaChartProps) {
  if (values.length === 0) {
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

  const fmt = formatValue ?? ((v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 }))

  const data = {
    labels,
    datasets: [
      {
        data: values,
        borderColor: color,
        backgroundColor: `${color}22`,
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: color,
      },
    ],
  }

  return (
    <div style={{ height }} aria-label={ariaLabel}>
      <Line
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              ...tooltipDefaults,
              callbacks: {
                label: (ctx) => ` ${ctx.parsed.y != null ? fmt(ctx.parsed.y) : '—'}`,
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
  )
}
