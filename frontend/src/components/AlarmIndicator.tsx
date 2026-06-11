interface AlarmIndicatorProps {
  priority: 0 | 1 | 2 | 3 | 4
  label?: string
  compact?: boolean
  className?: string
}

const priorityConfig = {
  4: { color: '#ff1744', shape: 'octagon', label: 'P4 Critical' },
  3: { color: '#ffd600', shape: 'triangle', label: 'P3 High' },
  2: { color: '#ff8c00', shape: 'square', label: 'P2 Advisory' },
  1: { color: '#ff00ff', shape: 'diamond', label: 'P1 Low' },
  0: { color: '#2979ff', shape: 'circle', label: 'P0 Diagnostic' },
} as const

function PriorityShape({
  priority,
  color,
}: {
  priority: 0 | 1 | 2 | 3 | 4
  color: string
}) {
  const cfg = priorityConfig[priority]
  const size = 22

  if (cfg.shape === 'octagon') {
    const pts = []
    const r = size / 2
    for (let i = 0; i < 8; i++) {
      const angle = (Math.PI / 4) * i - Math.PI / 8
      pts.push(`${r + r * 0.85 * Math.cos(angle)},${r + r * 0.85 * Math.sin(angle)}`)
    }
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <polygon points={pts.join(' ')} fill={color} />
        <text
          x="50%"
          y="50%"
          dominantBaseline="central"
          textAnchor="middle"
          fill="#fff"
          fontSize="14"
          fontWeight="700"
        >
          !
        </text>
      </svg>
    )
  }

  if (cfg.shape === 'triangle') {
    const cx = size / 2
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <polygon points={`${cx},2 ${size - 2},${size - 2} 2,${size - 2}`} fill={color} />
        <text
          x="50%"
          y="60%"
          dominantBaseline="central"
          textAnchor="middle"
          fill="#000"
          fontSize="13"
          fontWeight="700"
        >
          !
        </text>
      </svg>
    )
  }

  if (cfg.shape === 'square') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <rect x="2" y="4" width={size - 4} height={size - 4} rx="2" fill={color} />
      </svg>
    )
  }

  if (cfg.shape === 'diamond') {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <polygon
          points={`${size / 2},2 ${size - 2},${size / 2} ${size / 2},${size - 2} 2,${size / 2}`}
          fill={color}
        />
      </svg>
    )
  }

  // circle
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
      <circle cx={size / 2} cy={size / 2} r={size / 2 - 2} fill={color} />
    </svg>
  )
}

function SeverityBar({ priority, color }: { priority: number; color: string }) {
  const levels = [0, 1, 2, 3, 4]
  return (
    <div className="flex gap-0.5 items-center" aria-hidden="true">
      {levels.map((l) => (
        <div
          key={l}
          className={`w-2 h-3 rounded-sm ${
            l <= priority ? '' : 'bg-bg-hover'
          }`}
          style={l <= priority ? { backgroundColor: color } : undefined}
        />
      ))}
    </div>
  )
}

export default function AlarmIndicator({
  priority,
  label,
  compact = false,
  className = '',
}: AlarmIndicatorProps) {
  const cfg = priorityConfig[priority]
  const displayLabel = label ?? cfg.label

  if (compact) {
    return (
      <div className={`inline-flex items-center gap-2 ${className}`}>
        <SeverityBar priority={priority} color={cfg.color} />
        <span className="text-[11px] font-medium" style={{ color: cfg.color }}>
          {displayLabel}
        </span>
      </div>
    )
  }

  return (
    <div
      className={`inline-flex items-center gap-2 ${className}`}
      role="img"
      aria-label={`${cfg.label}: ${displayLabel}`}
    >
      <PriorityShape priority={priority} color={cfg.color} />
      <span className="text-xs font-medium" style={{ color: cfg.color }}>
        {displayLabel}
      </span>
    </div>
  )
}
