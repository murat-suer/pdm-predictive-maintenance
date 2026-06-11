import type { ReactNode } from 'react'

interface CardProps {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
}

export default function Card({
  title,
  subtitle,
  children,
  className = '',
}: CardProps) {
  return (
    <div
      className={`bg-bg-elevated border border-border-default rounded-lg p-4 ${className}`}
    >
      {title && (
        <div className="mb-3">
          <h3 className="text-text-primary text-sm font-medium m-0">{title}</h3>
          {subtitle && (
            <p className="text-text-tertiary text-xs mt-0.5 mb-0">
              {subtitle}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  )
}
