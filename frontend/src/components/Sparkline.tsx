interface SparklineProps {
  dataPoints: number[]
  color?: string
  height?: number
  className?: string
}

export default function Sparkline({
  dataPoints,
  color = '#448aff',
  height = 24,
  className = '',
}: SparklineProps) {
  if (dataPoints.length < 2) {
    return (
      <div className={`inline-flex items-center ${className}`} style={{ height }}>
        <span className="text-text-tertiary text-[10px]">—</span>
      </div>
    )
  }

  const width = dataPoints.length * 8
  const min = Math.min(...dataPoints)
  const max = Math.max(...dataPoints)
  const range = max - min || 1
  const padding = 2

  const points = dataPoints
    .map((v, i) => {
      const x = padding + (i / (dataPoints.length - 1)) * (width - padding * 2)
      const y = height - padding - ((v - min) / range) * (height - padding * 2)
      return `${x},${y}`
    })
    .join(' ')

  const areaPoints = `${padding},${height - padding} ${points} ${width - padding},${height - padding}`

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={`inline-block ${className}`}
      aria-hidden="true"
    >
      <polygon points={areaPoints} fill={`${color}15`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  )
}
