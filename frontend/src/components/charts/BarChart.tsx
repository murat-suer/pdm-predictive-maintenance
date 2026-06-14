import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Tooltip,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import { axisDefaults, tooltipDefaults } from './palette'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip)

interface BarChartProps {
  labels: string[]
  values: number[]
  colors: string[]
  ariaLabel: string
  horizontal?: boolean
  height?: number
  /** If provided, bar at this index gets a distinct highlight border */
  highlightIndex?: number
}

function fmt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export default function BarChart({
  labels,
  values,
  colors,
  ariaLabel,
  horizontal = false,
  height = 200,
  highlightIndex,
}: BarChartProps) {
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

  const borderWidths = values.map((_, i) => (i === highlightIndex ? 2 : 0))
  const borderColors = values.map((_, i) =>
    i === highlightIndex ? '#ffd600' : 'transparent',
  )

  const data = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: borderWidths,
        borderRadius: 3,
      },
    ],
  }

  const indexAxis = horizontal ? ('y' as const) : ('x' as const)

  return (
    <div style={{ height }} aria-label={ariaLabel}>
      <Bar
        data={data}
        options={{
          indexAxis,
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              ...tooltipDefaults,
              callbacks: {
                label: (ctx) => {
                  const val = ctx.parsed[horizontal ? 'x' : 'y']
                  return ` ${val != null ? fmt(val) : '—'}`
                },
              },
            },
          },
          scales: {
            x: {
              ...axisDefaults,
              ticks: {
                ...axisDefaults.ticks,
                maxRotation: horizontal ? 0 : 30,
                maxTicksLimit: horizontal ? 6 : undefined,
              },
            },
            y: {
              ...axisDefaults,
              ticks: {
                ...axisDefaults.ticks,
                maxTicksLimit: horizontal ? undefined : 5,
              },
            },
          },
        }}
      />
    </div>
  )
}
