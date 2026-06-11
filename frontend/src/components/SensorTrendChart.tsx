import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  TimeScale,
  Tooltip,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import type { SensorSeriesPoint } from '../api/types'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, TimeScale, Tooltip, Legend)

interface SensorTrendChartProps {
  sensorName: string
  unit: string | null
  points: SensorSeriesPoint[]
  warningThreshold: number | null
  criticalThreshold: number | null
  height?: number
}

export default function SensorTrendChart({
  sensorName,
  unit,
  points,
  warningThreshold,
  criticalThreshold,
  height = 160,
}: SensorTrendChartProps) {
  const labels = points.map((p) =>
    new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
  )

  const datasets = [
    {
      label: `${sensorName.replace(/_/g, ' ')}${unit ? ` (${unit})` : ''}`,
      data: points.map((p) => p.value),
      borderColor: '#60a5fa',
      backgroundColor: '#60a5fa',
      borderWidth: 1.5,
      tension: 0.25,
      // Anomalous readings show as red markers on the line.
      pointRadius: points.map((p) => (p.is_anomaly ? 3 : 0)),
      pointBackgroundColor: points.map((p) => (p.is_anomaly ? '#ff1744' : '#60a5fa')),
    },
    ...(warningThreshold != null
      ? [
          {
            label: 'warning',
            data: points.map(() => warningThreshold),
            borderColor: 'rgba(255, 214, 0, 0.5)',
            borderDash: [6, 4],
            borderWidth: 1,
            pointRadius: 0,
          },
        ]
      : []),
    ...(criticalThreshold != null
      ? [
          {
            label: 'critical',
            data: points.map(() => criticalThreshold),
            borderColor: 'rgba(255, 23, 68, 0.5)',
            borderDash: [6, 4],
            borderWidth: 1,
            pointRadius: 0,
          },
        ]
      : []),
  ]

  return (
    <div style={{ height }}>
      <Line
        data={{ labels, datasets }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#1a1a1f',
              titleFont: { family: "'IBM Plex Mono', monospace", size: 10 },
              bodyFont: { family: "'IBM Plex Mono', monospace", size: 10 },
            },
          },
          scales: {
            x: {
              ticks: {
                color: '#6b6b73',
                font: { family: "'IBM Plex Mono', monospace", size: 9 },
                maxTicksLimit: 6,
                maxRotation: 0,
              },
              grid: { color: 'rgba(255,255,255,0.04)' },
            },
            y: {
              ticks: {
                color: '#6b6b73',
                font: { family: "'IBM Plex Mono', monospace", size: 9 },
              },
              grid: { color: 'rgba(255,255,255,0.06)' },
            },
          },
        }}
      />
    </div>
  )
}
