import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  TimeScale,
  Tooltip,
  type ChartOptions,
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

// ISA-101 colorblind-safe:
//   - Anomalous points: crossRot shape (×), NOT just red color
//   - Warning threshold: dashed [6,4] + amber
//   - Critical threshold: dash-dot [6,3,2,3] + red (clearly distinct from warning)
// Text label annotation is added via a custom legend row below the chart.

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
      // Anomalous readings: crossRot (×) shape at radius 4 — shape distinguishes
      // without relying on color alone (ISA-101).
      pointRadius: points.map((p) => (p.is_anomaly ? 4 : 0)),
      pointStyle: points.map((p) => (p.is_anomaly ? ('crossRot' as const) : ('circle' as const))),
      pointBackgroundColor: points.map((p) => (p.is_anomaly ? '#ff1744' : '#60a5fa')),
      pointBorderColor: points.map((p) => (p.is_anomaly ? '#ff1744' : '#60a5fa')),
      pointBorderWidth: points.map((p) => (p.is_anomaly ? 2 : 1)),
    },
    ...(warningThreshold != null
      ? [
          {
            // Warning threshold: dashed line [6,4] — amber, dashed (ISA-18.2 yellow)
            label: `Warning (${warningThreshold})`,
            data: points.map(() => warningThreshold),
            borderColor: 'rgba(255, 214, 0, 0.7)',
            borderDash: [6, 4] as number[],
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0,
          },
        ]
      : []),
    ...(criticalThreshold != null
      ? [
          {
            // Critical threshold: dash-dot [6,3,2,3] — red, clearly distinct from warning
            label: `Critical (${criticalThreshold})`,
            data: points.map(() => criticalThreshold),
            borderColor: 'rgba(255, 23, 68, 0.75)',
            borderDash: [6, 3, 2, 3] as number[],
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0,
          },
        ]
      : []),
  ]

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      // Show a minimal legend so threshold labels appear as text, not just color
      legend: {
        display: warningThreshold != null || criticalThreshold != null,
        position: 'top',
        labels: {
          color: '#6b6b73',
          font: { family: "'IBM Plex Mono', monospace", size: 9 },
          boxWidth: 20,
          padding: 6,
          filter: (item) => item.datasetIndex !== 0, // hide main series from legend
          usePointStyle: false,
        },
      },
      tooltip: {
        backgroundColor: '#1a1a1f',
        titleFont: { family: "'IBM Plex Mono', monospace", size: 10 },
        bodyFont: { family: "'IBM Plex Mono', monospace", size: 10 },
        filter: (item) => item.datasetIndex === 0 || item.parsed.y != null,
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
  }

  return (
    <div>
      <div style={{ height }}>
        <Line data={{ labels, datasets }} options={options} />
      </div>
      {/* Redundant text legend for thresholds — colorblind-safe (ISA-101) */}
      {(warningThreshold != null || criticalThreshold != null) && (
        <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1">
          {warningThreshold != null && (
            <span className="flex items-center gap-1 text-[9px] text-text-tertiary">
              <svg width="20" height="6" viewBox="0 0 20 6" aria-hidden="true">
                <line
                  x1="0" y1="3" x2="20" y2="3"
                  stroke="rgba(255,214,0,0.7)"
                  strokeWidth="1.5"
                  strokeDasharray="6,4"
                />
              </svg>
              <span style={{ color: 'rgba(255,214,0,0.9)' }}>▲ Warning {warningThreshold}</span>
            </span>
          )}
          {criticalThreshold != null && (
            <span className="flex items-center gap-1 text-[9px] text-text-tertiary">
              <svg width="20" height="6" viewBox="0 0 20 6" aria-hidden="true">
                <line
                  x1="0" y1="3" x2="20" y2="3"
                  stroke="rgba(255,23,68,0.75)"
                  strokeWidth="1.5"
                  strokeDasharray="6,3,2,3"
                />
              </svg>
              <span style={{ color: 'rgba(255,23,68,0.9)' }}>■ Critical {criticalThreshold}</span>
            </span>
          )}
          {points.some((p) => p.is_anomaly) && (
            <span className="flex items-center gap-1 text-[9px]" style={{ color: '#ff1744' }}>
              <span aria-hidden="true">✕</span> Anomaly reading
            </span>
          )}
        </div>
      )}
    </div>
  )
}
