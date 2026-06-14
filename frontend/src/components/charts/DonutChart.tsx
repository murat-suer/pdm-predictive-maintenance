import { ArcElement, Chart as ChartJS, Tooltip } from 'chart.js'
import { Doughnut } from 'react-chartjs-2'
import { tooltipDefaults } from './palette'

ChartJS.register(ArcElement, Tooltip)

interface DonutSlice {
  label: string
  value: number
  color: string
}

interface DonutChartProps {
  slices: DonutSlice[]
  ariaLabel: string
  /** Render a custom HTML legend below the chart */
  showLegend?: boolean
  /** Inner radius cutout percentage (default 65) */
  cutout?: string
  height?: number
}

function fmt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export default function DonutChart({
  slices,
  ariaLabel,
  showLegend = true,
  cutout = '65%',
  height = 180,
}: DonutChartProps) {
  const total = slices.reduce((acc, s) => acc + s.value, 0)

  if (total === 0) {
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

  const data = {
    labels: slices.map((s) => s.label),
    datasets: [
      {
        data: slices.map((s) => s.value),
        backgroundColor: slices.map((s) => s.color),
        borderColor: '#16161b',
        borderWidth: 2,
        hoverBorderColor: '#23232a',
      },
    ],
  }

  return (
    <div aria-label={ariaLabel}>
      <div style={{ height }}>
        <Doughnut
          data={data}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            cutout,
            animation: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                ...tooltipDefaults,
                callbacks: {
                  label: (ctx) => {
                    const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : '0'
                    return ` ${ctx.label}: ${fmt(ctx.parsed)} (${pct}%)`
                  },
                },
              },
            },
          }}
        />
      </div>
      {showLegend && (
        <ul className="mt-2 space-y-1">
          {slices.map((s) => (
            <li key={s.label} className="flex items-center justify-between text-[11px]">
              <span className="flex items-center gap-1.5">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
                  style={{ backgroundColor: s.color }}
                />
                <span className="text-text-secondary">{s.label}</span>
              </span>
              <span className="font-mono text-text-primary tabular-nums">
                {fmt(s.value)}
                <span className="text-text-tertiary ml-1">
                  ({total > 0 ? ((s.value / total) * 100).toFixed(0) : 0}%)
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
