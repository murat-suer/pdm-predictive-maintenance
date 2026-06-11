import type { ReactNode } from 'react'

type PriorityVariant = 'p0' | 'p1' | 'p2' | 'p3' | 'p4'
type StatusVariant = 'success' | 'warning' | 'error' | 'info'
type BadgeVariant = PriorityVariant | StatusVariant

interface BadgeProps {
  variant: BadgeVariant
  label?: string
  children?: ReactNode
  className?: string
}

const shapeMap: Record<BadgeVariant, string> = {
  p4: '⬡',
  p3: '▲',
  p2: '■',
  p1: '◆',
  p0: '●',
  success: '●',
  warning: '▲',
  error: '⬡',
  info: '●',
}

const colorMap: Record<BadgeVariant, string> = {
  p4: 'text-alarm-p4 bg-alarm-p4/10 border-alarm-p4/30',
  p3: 'text-alarm-p3 bg-alarm-p3/10 border-alarm-p3/30',
  p2: 'text-alarm-p2 bg-alarm-p2/10 border-alarm-p2/30',
  p1: 'text-alarm-p1 bg-alarm-p1/10 border-alarm-p1/30',
  p0: 'text-alarm-p0 bg-alarm-p0/10 border-alarm-p0/30',
  success: 'text-success bg-success/10 border-success/30',
  warning: 'text-alarm-p3 bg-alarm-p3/10 border-alarm-p3/30',
  error: 'text-alarm-p4 bg-alarm-p4/10 border-alarm-p4/30',
  info: 'text-info bg-info/10 border-info/30',
}

export default function Badge({
  variant,
  label,
  children,
  className = '',
}: BadgeProps) {
  const displayLabel = label ?? (children as string | undefined) ?? variant

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-medium rounded-full border ${colorMap[variant]} ${className}`}
    >
      <span className="text-[10px] leading-none" aria-hidden="true">
        {shapeMap[variant]}
      </span>
      {displayLabel}
    </span>
  )
}
