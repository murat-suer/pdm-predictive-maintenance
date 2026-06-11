import { useState, useEffect } from 'react'

interface HeaderProps {
  title: string
  className?: string
}

export default function Header({ title, className = '' }: HeaderProps) {
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div
      className={`flex items-center justify-between border-b border-border-default pb-3 mb-4 ${className}`}
    >
      <div>
        <h1 className="text-xl font-semibold text-text-primary m-0">{title}</h1>
        <p className="text-text-tertiary text-xs mt-1 font-mono">
          {currentTime.toLocaleString('en-US', {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          })}
        </p>
      </div>
    </div>
  )
}
