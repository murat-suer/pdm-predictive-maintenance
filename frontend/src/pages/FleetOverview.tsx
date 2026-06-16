import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Header from '../components/Header'
import Sparkline from '../components/Sparkline'
import DonutChart from '../components/charts/DonutChart'
import MultiLineChart from '../components/charts/MultiLineChart'
import type { LineSeriesData } from '../components/charts/MultiLineChart'
import { PALETTE } from '../components/charts/palette'
import { useApi, useLiveSnapshot } from '../api/hooks'
import { useI18n, type TranslationKey } from '../i18n'
import type {
  FleetSummary,
  HealthTrendPoint,
  MachineSummary,
  MhiHistoryResponse,
} from '../api/types'

// ── ISA-101 redundant coding: color + shape + label per tier ────────────────
// Each tier has a DISTINCT shape so colorblind users never rely on hue alone.
const STATUS_COLOR: Record<string, string> = {
  normal:      'text-success',
  watch:       'text-alarm-p3',
  action:      'text-alarm-p2',
  critical:    'text-alarm-p4',
  maintenance: 'text-alarm-p0',
  offline:     'text-text-tertiary',
  // legacy alias kept for safety
  warning:     'text-alarm-p3',
}

/** Unicode glyphs — each tier has a visually distinct shape */
const STATUS_SHAPE: Record<string, string> = {
  normal:      '●',   // filled circle
  watch:       '▲',   // triangle
  action:      '◆',   // diamond
  critical:    '■',   // square (filled)
  maintenance: '▣',   // square with inner square
  offline:     '○',   // hollow circle
  // legacy alias
  warning:     '▲',
}

// ISA-101: green/teal good, amber watch, orange action, red critical, blue maintenance, gray offline
const STATUS_DONUT_COLORS: Record<string, string> = {
  normal:      PALETTE.GOOD,
  watch:       PALETTE.ALARM_P3,
  action:      '#fb923c',         // orange-400 — between amber and red
  critical:    PALETTE.ALARM_P4,
  maintenance: PALETTE.OBSERVE,
  offline:     PALETTE.TICK,
  warning:     PALETTE.ALARM_P3,
}

// Donut legend also shows shape so it is redundant-coded
const STATUS_LEGEND_SHAPE: Record<string, string> = STATUS_SHAPE

const machineTypes = ['All', 'Compressor', 'Heat Exchanger', 'Conveyor'] as const

function fmtLabel(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Compact per-machine health block — health bar + health%, RUL, status badge, button (ISA-101)
function HealthBar({
  id,
  score,
  rul,
  status,
  navigate,
}: {
  id: string
  score: number | null
  rul: number | null
  status: string
  navigate: ReturnType<typeof useNavigate>
}) {
  const { t } = useI18n()
  const pct = score != null ? Math.min(Math.max(score * 100, 0), 100) : null
  // ISA-101: red is reserved for genuinely critical states. The bar length
  // shows the health magnitude; its COLOUR follows the machine's status tier
  // (the recommendation), so a low-MHI machine the engine still rates Normal
  // stays green instead of alarming red.
  // Use the hex palette (STATUS_DONUT_COLORS), not the Tailwind class map
  // (STATUS_COLOR) — this value is an inline backgroundColor.
  const barColor = pct == null ? PALETTE.TICK : (STATUS_DONUT_COLORS[status] ?? PALETTE.GOOD)
  const rulDisplay = rul == null ? '—' : rul < 1 ? '<1' : rul.toFixed(0)

  return (
    <div className="flex flex-col gap-1 min-w-0">
      {/* Machine ID */}
      <span className="text-[10px] font-mono text-text-secondary truncate" title={id}>
        {id}
      </span>

      {/* Health bar + % */}
      <div className="flex items-center gap-1.5 min-w-0">
        <div className="flex-1 relative h-2.5 bg-bg-secondary rounded-sm overflow-hidden border border-border-subtle">
          {pct != null ? (
            <div
              className="absolute left-0 top-0 h-full rounded-sm transition-all duration-500"
              style={{ width: `${pct}%`, backgroundColor: barColor, opacity: 0.75 }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[8px] text-text-tertiary">—</span>
            </div>
          )}
        </div>
        <span
          className="text-[10px] font-mono tabular-nums shrink-0 w-7 text-right"
          style={{ color: barColor }}
        >
          {pct != null ? `${pct.toFixed(0)}%` : '—'}
        </span>
      </div>

      {/* RUL + status badge */}
      <div className="flex items-center justify-between gap-1">
        <span className="text-[10px] text-text-tertiary">
          RUL: <span className="font-mono text-text-secondary">{rulDisplay} h</span>
        </span>
        <span className={`text-[10px] font-medium ${STATUS_COLOR[status] ?? 'text-text-tertiary'} flex items-center gap-0.5 shrink-0`}>
          <span aria-hidden="true">{STATUS_SHAPE[status] ?? '○'}</span>
          <span>{t(`status.${status}` as TranslationKey)}</span>
        </span>
      </div>

      {/* Navigate button */}
      <Button
        variant="secondary"
        size="sm"
        className="w-full text-[10px] px-2 py-0.5 min-h-0 h-6"
        onClick={() => navigate(`/machines/${id}`)}
        title={t('fleet.openMachine')}
      >
        {t('fleet.openMachineLink')} ↗
      </Button>
    </div>
  )
}

export default function FleetOverview() {
  const [filter, setFilter] = useState<'All' | 'Alert' | 'Normal'>('All')
  const [typeFilter, setTypeFilter] = useState<string>('All')
  const navigate = useNavigate()
  const { t } = useI18n()

  const machinesApi = useApi<MachineSummary[]>('/machines', 15000)
  const summaryApi = useApi<FleetSummary>('/fleet/summary', 15000)
  const trendApi = useApi<HealthTrendPoint[]>('/fleet/health-trend?hours=24', 60000)
  const mhiHistoryApi = useApi<MhiHistoryResponse>('/analytics/mhi-history?buckets=24', 60000)
  const { snapshot } = useLiveSnapshot()

  // Overlay live WebSocket snapshot onto the REST machine list.
  const machines = useMemo(() => {
    const base = machinesApi.data ?? []
    if (!snapshot) return base
    const liveById = new Map(snapshot.machines.map((m) => [m.id, m]))
    return base.map((m) => {
      const live = liveById.get(m.id)
      return live
        ? {
            ...m,
            status: live.status,
            top_alarm: live.top_alarm,
            health_score: live.health_score,
            rul_hours: live.rul_hours ?? m.rul_hours,
            reliability: live.reliability ?? m.reliability,
          }
        : m
    })
  }, [machinesApi.data, snapshot])

  // Task 2: "Alert" filter means any of watch/action/critical (not just !normal)
  const filteredMachines = machines.filter((m) => {
    const isAlert = m.status === 'watch' || m.status === 'action' || m.status === 'critical' || m.status === 'warning'
    if (filter === 'Alert' && !isAlert) return false
    if (filter === 'Normal' && m.status !== 'normal') return false
    if (typeFilter !== 'All' && m.type !== typeFilter) return false
    return true
  })

  const summary = summaryApi.data
  const trend = trendApi.data ?? []

  // ── Fleet status donut slices — shape+text in legend (Task 3) ─────────────
  const donutSlices = useMemo(() => {
    if (!summary) return []
    const entries: { key: string; label: string; count: number }[] = [
      { key: 'normal',      label: `${STATUS_LEGEND_SHAPE['normal']} ${t('status.normal')}`,      count: summary.normal },
      { key: 'watch',       label: `${STATUS_LEGEND_SHAPE['watch']} ${t('status.watch')}`,         count: summary.watch ?? 0 },
      { key: 'action',      label: `${STATUS_LEGEND_SHAPE['action']} ${t('status.action')}`,       count: summary.action ?? 0 },
      { key: 'critical',    label: `${STATUS_LEGEND_SHAPE['critical']} ${t('status.critical')}`,   count: summary.critical },
      { key: 'maintenance', label: `${STATUS_LEGEND_SHAPE['maintenance']} ${t('fleet.maintenance')}`, count: summary.maintenance },
    ]
    return entries
      .filter((e) => e.count > 0)
      .map((e) => ({
        label: e.label,
        value: e.count,
        color: STATUS_DONUT_COLORS[e.key] ?? PALETTE.NEUTRAL,
      }))
  }, [summary, t])

  // ── MHI multi-line chart data ──────────────────────────────────────────────
  const mhiMachines = useMemo(
    () => mhiHistoryApi.data?.machines ?? [],
    [mhiHistoryApi.data],
  )
  const mhiAllTs = useMemo(
    () =>
      Array.from(new Set(mhiMachines.flatMap((m) => m.points.map((p) => p.t)))).sort(),
    [mhiMachines],
  )
  const mhiLabels = mhiAllTs.map((ts) => fmtLabel(ts))
  const mhiSeries: LineSeriesData[] = useMemo(
    () =>
      mhiMachines.map((machine, idx) => {
        const byTime = new Map(machine.points.map((p) => [p.t, p.mhi]))
        return {
          id: machine.machine_id,
          label: machine.machine_id,
          values: mhiAllTs.map((ts) => byTime.get(ts) ?? NaN),
          color: PALETTE.MACHINE_LINES[idx % PALETTE.MACHINE_LINES.length],
        }
      }),
    [mhiMachines, mhiAllTs],
  )

  if (machinesApi.error && machines.length === 0) {
    return (
      <div className="p-4">
        <Header title={t('fleet.title')} />
        <Card>
          <p className="text-alarm-p4 text-sm">
            {t('fleet.backendUnreachable')} {machinesApi.error}
          </p>
          <p className="text-text-tertiary text-xs mt-1">{t('fleet.checkApi')}</p>
        </Card>
      </div>
    )
  }

  const filterLabel: Record<'All' | 'Alert' | 'Normal', string> = {
    All: t('fleet.filter.all'),
    Alert: t('fleet.filter.alert'),
    Normal: t('fleet.filter.normal'),
  }

  return (
    <div className="p-4">
      <Header title={t('fleet.title')} />

      {/* 6 KPI tier cards — Online · Normal · Dikkat · Uyarı · Kritik · Bakımda (ISA-101: shape+colour+text) */}
      <div className="flex flex-wrap gap-3 mb-4">
        {/* Çevrimiçi — total − offline, no per-machine tier colour (it's a count) */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-text-tertiary text-[11px] uppercase tracking-wider">
              {t('fleet.online')}
            </span>
            <span className="text-text-primary text-2xl font-semibold font-mono">
              {summary ? summary.online : '—'}
            </span>
          </div>
        </Card>
        {/* Normal ● */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-success text-[11px] uppercase tracking-wider flex items-center gap-1">
              <span aria-hidden="true">{STATUS_SHAPE.normal}</span>
              {t('status.normal')}
            </span>
            <span className="text-success text-2xl font-semibold font-mono">
              {summary ? summary.normal : '—'}
            </span>
          </div>
        </Card>
        {/* Watch ▲ */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-alarm-p3 text-[11px] uppercase tracking-wider flex items-center gap-1">
              <span aria-hidden="true">{STATUS_SHAPE.watch}</span>
              {t('status.watch')}
            </span>
            <span className="text-alarm-p3 text-2xl font-semibold font-mono">
              {summary ? (summary.watch ?? 0) : '—'}
            </span>
          </div>
        </Card>
        {/* Action ◆ */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-[11px] uppercase tracking-wider flex items-center gap-1" style={{ color: STATUS_DONUT_COLORS.action }}>
              <span aria-hidden="true">{STATUS_SHAPE.action}</span>
              {t('status.action')}
            </span>
            <span className="text-2xl font-semibold font-mono" style={{ color: STATUS_DONUT_COLORS.action }}>
              {summary ? (summary.action ?? 0) : '—'}
            </span>
          </div>
        </Card>
        {/* Critical ■ */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-alarm-p4 text-[11px] uppercase tracking-wider flex items-center gap-1">
              <span aria-hidden="true">{STATUS_SHAPE.critical}</span>
              {t('status.critical')}
            </span>
            <span className="text-alarm-p4 text-2xl font-semibold font-mono">
              {summary ? summary.critical : '—'}
            </span>
          </div>
        </Card>
        {/* Maintenance ▣ */}
        <Card className="min-w-[110px] flex-1">
          <div className="flex flex-col">
            <span className="text-alarm-p0 text-[11px] uppercase tracking-wider flex items-center gap-1">
              <span aria-hidden="true">{STATUS_SHAPE.maintenance}</span>
              {t('status.maintenance')}
            </span>
            <span className="text-alarm-p0 text-2xl font-semibold font-mono">
              {summary ? summary.maintenance : '—'}
            </span>
          </div>
        </Card>
      </div>

      {/* ── Visual band: Status Donut + MHI Trend ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        {/* Fleet status donut — shape+text in legend (ISA-101 Task 3) */}
        <Card title={t('fleet.statusDist')} subtitle={t('fleet.statusDistSub')}>
          {summaryApi.loading && !summary ? (
            <div className="flex items-center justify-center h-40 text-text-tertiary text-xs">
              {t('fleet.loading')}
            </div>
          ) : (
            <DonutChart
              slices={donutSlices}
              ariaLabel={t('fleet.statusDist')}
              height={160}
              showLegend
            />
          )}
        </Card>

        {/* Per-machine MHI multi-line trend — 2/3 of the row */}
        <Card
          title={t('fleet.mhiTrend')}
          subtitle={t('fleet.mhiTrendSub')}
          className="md:col-span-2"
        >
          {mhiHistoryApi.loading && mhiSeries.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-text-tertiary text-xs">
              {t('fleet.loading')}
            </div>
          ) : mhiSeries.length === 0 ? (
            <p className="text-text-tertiary text-xs py-6 text-center">
              {t('fleet.noHealthHistory')}
            </p>
          ) : (
            <MultiLineChart
              labels={mhiLabels}
              series={mhiSeries}
              ariaLabel={t('fleet.mhiTrend')}
              height={160}
              yMin={0}
              yMax={100}
              formatValue={(v) => v.toFixed(0)}
            />
          )}
        </Card>
      </div>

      {/* ── Health Snapshot: per-machine health/RUL/status/button (ISA-101) ── */}
      {machines.length > 0 && (
        <Card title={t('fleet.healthSnapshot')} subtitle={t('fleet.healthSnapshotSub')} className="mb-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-x-4 gap-y-4 pt-1">
            {machines.map((m) => (
              <HealthBar
                key={m.id}
                id={m.id}
                score={m.health_score}
                rul={m.rul_hours}
                status={m.status}
                navigate={navigate}
              />
            ))}
          </div>
        </Card>
      )}

      {/* Fleet Health Trend (aggregate legacy bar) */}
      <Card title={t('fleet.healthTrend')} subtitle={t('fleet.healthTrendSub')} className="mb-4">
        {trend.length === 0 ? (
          <p className="text-text-tertiary text-xs py-6 text-center">{t('fleet.noHealthHistory')}</p>
        ) : (
          <>
            <div className="flex items-end gap-0.5 h-24">
              {trend.map((point) => {
                const value = point.avg_health_score * 100
                return (
                  <div
                    key={point.bucket}
                    className="flex-1 rounded-t-sm transition-all duration-300"
                    style={{
                      height: `${Math.max(value, 2)}%`,
                      // Aggregate trend is a metric, not an alarm — keep red
                      // reserved: amber when the fleet mean dips, blue otherwise.
                      backgroundColor: value < 70 ? '#ffd600' : '#60a5fa',
                      opacity: 0.7,
                    }}
                    title={`${new Date(point.bucket).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} — MHI ${value.toFixed(0)}`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between text-[10px] text-text-tertiary mt-1">
              <span>-24h</span>
              <span>{t('fleet.now')}</span>
            </div>
          </>
        )}
      </Card>

      {/* Filters */}
      <div className="flex items-center gap-2 mb-3">
        <div className="flex gap-1 bg-bg-secondary rounded-md border border-border-default p-0.5">
          {(['All', 'Alert', 'Normal'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                filter === f
                  ? 'bg-bg-elevated text-text-primary border border-border-default'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {filterLabel[f]}
            </button>
          ))}
        </div>

        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="bg-bg-secondary text-text-secondary text-xs border border-border-default rounded px-2 py-1 hover:text-text-primary focus:outline-none focus:border-alarm-p0"
        >
          {machineTypes.map((mt) => (
            <option key={mt} value={mt}>
              {mt === 'All' ? t('fleet.filter.all') : mt}
            </option>
          ))}
        </select>
      </div>

      {/* Machine Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-text-tertiary border-b border-border-default">
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.machine')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.type')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.line')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.status')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.rul')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.health')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.reliability')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.topAlarm')}</th>
                <th className="pb-2 pr-4 font-medium">{t('fleet.col.trend')}</th>
              </tr>
            </thead>
            <tbody>
              {filteredMachines.map((m) => (
                <tr
                  key={m.id}
                  onClick={() => navigate(`/machines/${m.id}`)}
                  title={t('fleet.openMachine')}
                  className="border-b border-border-subtle last:border-0 cursor-pointer hover:bg-bg-hover transition-colors"
                >
                  <td className="py-2.5 pr-4">
                    <Link
                      to={`/machines/${m.id}`}
                      onClick={(e) => e.stopPropagation()}
                      className="text-alarm-p0 font-medium hover:underline transition-colors"
                    >
                      {m.id} ↗
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4 text-text-secondary">{m.type}</td>
                  <td className="py-2.5 pr-4 text-text-secondary">{m.line}</td>
                  {/* Task 2: Status cell always shape + color + text label */}
                  <td className="py-2.5 pr-4">
                    <span className={`${STATUS_COLOR[m.status] ?? 'text-text-tertiary'} flex items-center gap-1`}>
                      <span aria-hidden="true">{STATUS_SHAPE[m.status] ?? '○'}</span>
                      <span>{t(`status.${m.status}` as TranslationKey)}</span>
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-text-secondary">
                    {m.rul_hours == null ? '—' : m.rul_hours < 1 ? '<1' : m.rul_hours.toFixed(0)}
                  </td>
                  <td className="py-2.5 pr-4 font-mono">
                    {m.health_score == null ? (
                      <span className="text-text-tertiary">—</span>
                    ) : (
                      // Health % is a metric, not a priority signal — leave it
                      // neutral; the Status column carries the (red-reserved)
                      // tier colour.
                      <span className="text-text-primary">
                        {(m.health_score * 100).toFixed(0)}%
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 pr-4 font-mono">
                    <span
                      className={
                        m.reliability == null
                          ? 'text-text-tertiary'
                          : m.reliability < 50
                            ? 'text-alarm-p4'
                            : m.reliability < 80
                              ? 'text-alarm-p3'
                              : 'text-text-primary'
                      }
                    >
                      {m.reliability != null ? `${m.reliability.toFixed(1)}%` : '—'}
                    </span>
                  </td>
                  <td className="py-2.5 pr-4">
                    {m.top_alarm ? (
                      <Badge variant="warning" label={m.top_alarm.replace(/_/g, ' ')} />
                    ) : (
                      <span className="text-text-tertiary">—</span>
                    )}
                  </td>
                  <td className="py-2.5">
                    {m.health_history.length > 1 ? (
                      <Sparkline
                        dataPoints={m.health_history}
                        color={
                          m.status === 'critical'
                            ? '#ff1744'
                            : m.status === 'watch' || m.status === 'warning'
                              ? '#ffd600'
                              : m.status === 'action'
                                ? '#fb923c'
                                : '#60a5fa'
                        }
                        height={20}
                      />
                    ) : (
                      <span className="text-text-tertiary">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!machinesApi.loading && filteredMachines.length === 0 && (
            <p className="text-text-tertiary text-center py-6">{t('fleet.noMatch')}</p>
          )}
          {machinesApi.loading && (
            <p className="text-text-tertiary text-center py-6">{t('fleet.loading')}</p>
          )}
        </div>
      </Card>
    </div>
  )
}
