import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Header from '../components/Header'
import Sparkline from '../components/Sparkline'
import { useApi, useLiveSnapshot } from '../api/hooks'
import { useI18n, type TranslationKey } from '../i18n'
import type { FleetSummary, HealthTrendPoint, MachineSummary, SavingsSummary } from '../api/types'

const statusColor: Record<string, string> = {
  normal: 'text-success',
  warning: 'text-alarm-p3',
  critical: 'text-alarm-p4',
  maintenance: 'text-alarm-p0',
  offline: 'text-text-tertiary',
}

const statusShape: Record<string, string> = {
  normal: '●',
  warning: '▲',
  critical: '⬡',
  maintenance: '⚙',
  offline: '—',
}

const machineTypes = ['All', 'Compressor', 'Heat Exchanger', 'Conveyor'] as const

export default function FleetOverview() {
  const [filter, setFilter] = useState<'All' | 'Alert' | 'Normal'>('All')
  const [typeFilter, setTypeFilter] = useState<string>('All')
  const navigate = useNavigate()
  const { t, lang } = useI18n()

  const machinesApi = useApi<MachineSummary[]>('/machines', 15000)
  const summaryApi = useApi<FleetSummary>('/fleet/summary', 15000)
  const trendApi = useApi<HealthTrendPoint[]>('/fleet/health-trend?hours=24', 60000)
  const savingsApi = useApi<SavingsSummary>('/savings', 30000)
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
            // Never let a sparser live frame blank out values the REST
            // endpoint already resolved.
            rul_hours: live.rul_hours ?? m.rul_hours,
            reliability: live.reliability ?? m.reliability,
          }
        : m
    })
  }, [machinesApi.data, snapshot])

  const filteredMachines = machines.filter((m) => {
    if (filter === 'Alert' && m.status === 'normal') return false
    if (filter === 'Normal' && m.status !== 'normal') return false
    if (typeFilter !== 'All' && m.type !== typeFilter) return false
    return true
  })

  const summary = summaryApi.data
  const trend = trendApi.data ?? []

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

      {/* KPI Cards */}
      <div className="grid grid-cols-5 gap-3 mb-4">
        <Card>
          <div className="flex flex-col">
            <span className="text-text-tertiary text-[11px] uppercase tracking-wider">
              {t('fleet.online')}
            </span>
            <span className="text-text-primary text-2xl font-semibold font-mono">
              {summary ? summary.normal : '—'}
            </span>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-text-tertiary text-[11px] uppercase tracking-wider">
              {t('fleet.warning')}
            </span>
            <span className="text-alarm-p3 text-2xl font-semibold font-mono">
              {summary ? summary.warning : '—'}
            </span>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-text-tertiary text-[11px] uppercase tracking-wider">
              {t('fleet.critical')}
            </span>
            <span className="text-alarm-p4 text-2xl font-semibold font-mono">
              {summary ? summary.critical : '—'}
            </span>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span className="text-text-tertiary text-[11px] uppercase tracking-wider">
              {t('fleet.maintenance')}
            </span>
            <span className="text-alarm-p0 text-2xl font-semibold font-mono">
              {summary ? summary.maintenance : '—'}
            </span>
          </div>
        </Card>
        <Card>
          <div className="flex flex-col">
            <span
              className="text-text-tertiary text-[11px] uppercase tracking-wider"
              title={t('fleet.savingsTooltip')}
            >
              {t('fleet.savings')}
              {savingsApi.data && (
                <span className="normal-case tracking-normal">
                  {' '}
                  · {t('fleet.savingsWindow', { hours: savingsApi.data.window_hours })}
                </span>
              )}
            </span>
            <span className="text-success text-2xl font-semibold font-mono">
              {savingsApi.data
                ? `€${Math.max(savingsApi.data.total_savings_eur, 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                : '—'}
            </span>
            <span className="text-text-tertiary text-[10px]">
              {savingsApi.data
                ? `${savingsApi.data.maintenance_count} ${t('fleet.maintenanceEvents')}`
                : ''}
            </span>
          </div>
        </Card>
      </div>

      {/* Fleet Health Trend */}
      <Card title={t('fleet.healthTrend')} subtitle={t('fleet.healthTrendSub')} className="mb-4">
        {trend.length === 0 ? (
          <p className="text-text-tertiary text-xs py-6 text-center">{t('fleet.noHealthHistory')}</p>
        ) : (
          <>
            <div className="flex items-end gap-0.5 h-24">
              {trend.map((point) => {
                const value = point.avg_health_score * 100
                // ISA-101 color discipline: the trend reads neutral by
                // default; yellow/red only at the MHI classification
                // boundaries (Degrading <70, Critical <55) — red is
                // reserved for genuinely abnormal fleet states.
                return (
                  <div
                    key={point.bucket}
                    className="flex-1 rounded-t-sm transition-all duration-300"
                    style={{
                      height: `${Math.max(value, 2)}%`,
                      backgroundColor: value < 55 ? '#ff1744' : value < 70 ? '#ffd600' : '#60a5fa',
                      opacity: 0.7,
                    }}
                    title={`${new Date(point.bucket).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} — MHI ${value.toFixed(0)}`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between text-[10px] text-text-tertiary mt-1">
              <span>-24h</span>
              <span>{lang === 'tr' ? 'Şimdi' : 'Now'}</span>
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
                  <td className="py-2.5 pr-4">
                    <span className={`${statusColor[m.status]} flex items-center gap-1`}>
                      <span aria-hidden="true">{statusShape[m.status]}</span>
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
                      <span
                        className={
                          m.health_score < 0.55
                            ? 'text-alarm-p4'
                            : m.health_score < 0.7
                              ? 'text-alarm-p3'
                              : 'text-text-primary'
                        }
                      >
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
                        color={m.status === 'critical' ? '#ff1744' : m.status === 'warning' ? '#ffd600' : '#60a5fa'}
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
