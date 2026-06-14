/**
 * ISA-101 HMI colour palette for Chart.js canvas contexts.
 * Canvas cannot read CSS custom properties, so we centralise colours here.
 * Red ONLY for alarm/shutdown, amber for warning, teal/green for good,
 * blue/gray neutral. No rainbow.
 */

export const PALETTE = {
  // Scenario / decision colours
  OBSERVE: '#60a5fa',          // blue-400 — neutral observation
  PLANNED: '#2dd4bf',          // teal-400 — positive, scheduled action
  DISPATCH_TECHNICIAN: '#a78bfa', // violet-400 — active intervention (neutral-positive)
  REDUCE_LOAD: '#fb923c',      // orange-400 — caution (between warning and action)
  SHUTDOWN: '#ff1744',         // ISA-101 alarm-P4 red — emergency

  // ISA-101 semantic
  ALARM_P4: '#ff1744',         // critical / shutdown
  ALARM_P3: '#ffd600',         // warning / amber
  GOOD: '#2dd4bf',             // savings / positive / teal
  NEUTRAL: '#60a5fa',          // blue — normal / info

  // Grays
  GRID: 'rgba(255,255,255,0.06)',
  GRID_SUBTLE: 'rgba(255,255,255,0.04)',
  TICK: '#6b6b73',
  TOOLTIP_BG: '#1a1a1f',

  // Multi-line machine colours (up to 8, no red/yellow — those are reserved for alarms)
  MACHINE_LINES: [
    '#60a5fa', // blue-400
    '#2dd4bf', // teal-400
    '#a78bfa', // violet-400
    '#34d399', // emerald-400
    '#818cf8', // indigo-400
    '#38bdf8', // sky-400
    '#4ade80', // green-400
    '#c084fc', // purple-400
  ],
} as const

export const CHART_FONT = "'IBM Plex Mono', monospace"

export const axisDefaults = {
  ticks: { color: PALETTE.TICK, font: { family: CHART_FONT, size: 9 } },
  grid: { color: PALETTE.GRID },
}

export const tooltipDefaults = {
  backgroundColor: PALETTE.TOOLTIP_BG,
  titleFont: { family: CHART_FONT, size: 10 },
  bodyFont: { family: CHART_FONT, size: 10 },
}
