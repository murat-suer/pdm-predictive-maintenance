import { useRef, useEffect } from 'react'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  type ChartData,
  type TooltipItem,
} from 'chart.js'
import { Radar } from 'react-chartjs-2'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

interface RadarVariable {
  name: string
  value: number
  baseline: number
  unit?: string
}

interface RadarPlotProps {
  variables: RadarVariable[]
  size?: number
  className?: string
}

export default function RadarPlot({
  variables,
  size = 280,
  className = '',
}: RadarPlotProps) {
  const chartRef = useRef<ChartJS<'radar'>>(null)

  const alarmThreshold = 1.3 // 30% above baseline triggers alarm edge

  const labels = variables.map((v) => v.name)
  const values = variables.map((v) => v.value)
  const baselines = variables.map((v) => v.baseline)

  // Determine if any variable crosses threshold
  const hasAlarm = variables.some(
    (v) => v.value > v.baseline * alarmThreshold || v.value < v.baseline * (1 / alarmThreshold),
  )

  const borderColor = hasAlarm
    ? variables.map((v) =>
        v.value > v.baseline * alarmThreshold || v.value < v.baseline * (1 / alarmThreshold)
          ? '#ff1744'
          : '#448aff',
      )
    : '#448aff'

  const data: ChartData<'radar'> = {
    labels,
    datasets: [
      {
        label: 'Current',
        data: values,
        backgroundColor: 'rgba(68, 138, 255, 0.15)',
        borderColor: borderColor,
        borderWidth: borderColor === '#448aff' ? 2 : 2,
        pointBackgroundColor: borderColor,
        pointBorderColor: borderColor,
        pointRadius: 3,
        pointHoverRadius: 5,
      },
      {
        label: 'Baseline',
        data: baselines,
        backgroundColor: 'rgba(107, 107, 115, 0.05)',
        borderColor: 'rgba(107, 107, 115, 0.4)',
        borderWidth: 1,
        borderDash: [4, 4],
        pointBackgroundColor: 'rgba(107, 107, 115, 0.3)',
        pointBorderColor: 'transparent',
        pointRadius: 0,
        pointHoverRadius: 0,
      },
    ],
  }

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.update()
    }
  }, [variables])

  return (
    <div className={`inline-flex flex-col items-center ${className}`}>
      <Radar
        ref={chartRef}
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              display: true,
              labels: {
                color: '#a0a0a8',
                font: { family: "'IBM Plex Sans', sans-serif", size: 10 },
                boxWidth: 12,
                padding: 8,
              },
            },
            tooltip: {
              callbacks: {
                label: (ctx: TooltipItem<'radar'>) => {
                  const idx = ctx.dataIndex
                  const v = variables[idx]
                  return `${v.name}: ${v.value}${v.unit ?? ''} (baseline: ${v.baseline}${v.unit ?? ''})`
                },
              },
            },
          },
          scales: {
            r: {
              beginAtZero: true,
              grid: { color: 'rgba(255,255,255,0.06)' },
              angleLines: { color: 'rgba(255,255,255,0.06)' },
              pointLabels: {
                color: '#a0a0a8',
                font: { family: "'IBM Plex Sans', sans-serif", size: 10 },
              },
              ticks: {
                color: '#6b6b73',
                font: { family: "'IBM Plex Sans', sans-serif", size: 9 },
                backdropColor: 'transparent',
                stepSize: 25,
              },
            },
          },
        }}
        width={size}
        height={size}
      />
    </div>
  )
}
