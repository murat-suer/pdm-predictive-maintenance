import Sparkline from './Sparkline'

interface GaugeProps {
  label: string
  value: number
  min: number
  max: number
  normalRange: [number, number]
  warningRange: [number, number]
  criticalRange: [number, number]
  unit?: string
  dataPoints?: number[]
  className?: string
}

function trendArrow(dataPoints?: number[]): '↑' | '↓' | '→' {
  if (!dataPoints || dataPoints.length < 2) return '→'
  const last = dataPoints[dataPoints.length - 1]
  const prev = dataPoints[dataPoints.length - 2]
  if (last > prev) return '↑'
  if (last < prev) return '↓'
  return '→'
}

function rangeColor(
  value: number,
  normalRange: [number, number],
  warningRange: [number, number],
  criticalRange: [number, number],
): string {
  if (value >= criticalRange[0] && value <= criticalRange[1]) return '#ff1744'
  if (value >= warningRange[0] && value <= warningRange[1]) return '#ffd600'
  if (value >= normalRange[0] && value <= normalRange[1]) return '#60a5fa'
  return '#6b6b73'
}

export default function Gauge({
  label,
  value,
  min,
  max,
  normalRange,
  warningRange,
  criticalRange,
  unit = '',
  dataPoints,
  className = '',
}: GaugeProps) {
  const pct = ((value - min) / (max - min)) * 100
  const color = rangeColor(value, normalRange, warningRange, criticalRange)
  const arrow = trendArrow(dataPoints)

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-primary font-medium truncate">{label}</span>
        <span className="text-text-secondary font-mono tabular-nums">
          {value.toFixed(1)}
          {unit && <span className="ml-0.5">{unit}</span>}
          <span className="ml-1 text-sm">{arrow}</span>
        </span>
      </div>

      <div className="relative h-4 w-full bg-bg-secondary rounded-sm overflow-hidden border border-border-subtle">
        <div
          className="absolute left-0 top-0 h-full rounded-sm"
          style={{
            width: `${Math.min(pct, 100)}%`,
            backgroundColor: color,
            opacity: 0.7,
          }}
        />
        <div
          className="absolute top-0 w-1 h-full rounded-sm"
          style={{
            left: `${Math.min(pct, 100)}%`,
            backgroundColor: color,
            transform: 'translateX(-50%)',
          }}
        />
      </div>

      <div className="flex justify-between text-[10px] text-text-tertiary">
        <span>{min}</span>
        <span>{max}</span>
      </div>

      {dataPoints && dataPoints.length >= 2 && (
        <Sparkline dataPoints={dataPoints} color={color} height={16} />
      )}
    </div>
  )
}
