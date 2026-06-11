import type { ButtonHTMLAttributes, ReactNode } from 'react'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'alarm'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  children: ReactNode
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    'bg-[#3b82f6] text-white border-[#3b82f6] hover:bg-[#2563eb] active:bg-[#1d4ed8]',
  secondary:
    'bg-transparent text-text-primary border-border-default hover:bg-bg-hover active:bg-bg-elevated',
  ghost:
    'bg-transparent text-text-secondary border-transparent hover:text-text-primary hover:bg-bg-hover active:bg-bg-elevated',
  alarm:
    'bg-alarm-p4 text-white border-alarm-p4 hover:brightness-110 active:brightness-90',
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs min-h-8',
  md: 'px-4 py-2 text-sm min-h-10',
  lg: 'px-6 py-3 text-base min-h-14',
}

export default function Button({
  variant = 'primary',
  size = 'md',
  className = '',
  children,
  ...props
}: ButtonProps) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-md border font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-alarm-p0/50 disabled:opacity-40 disabled:pointer-events-none select-none'

  return (
    <button
      className={`${base} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
