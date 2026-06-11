import { useState } from 'react'
import { useParams } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Gauge from '../components/Gauge'
import RadarPlot from '../components/RadarPlot'
import Header from '../components/Header'
import SensorTrendChart from '../components/SensorTrendChart'
import { apiPost } from '../api/client'
import { useApi } from '../api/hooks'
import { useI18n, type TranslationKey } from '../i18n'
import type {
  AlarmItem,
  MachineDetailData,
  SensorSeries,
  SensorSnapshot,
  WhatIfResult,
  WorkOrderItem,
} from '../api/types'

interface GaugeRanges {
  min: number
  max: number
  normalRange: [number, number]
  warningRange: [number, number]
  criticalRange: [number, number]
}

/** Derive gauge bands from warning/critical thresholds and degradation direction. */
function gaugeRanges(sensor: SensorSnapshot): GaugeRanges {
  const warn = sensor.warning_threshold ?? 0
  const crit = sensor.critical_threshold ?? 0
  const nominal = sensor.nominal_mu ?? 0
  if ((sensor.degradation_direction ?? 1) >= 0) {
    // Higher is worse (vibration, temperature)
    const max = Math.max(crit * 1.3, sensor.value ?? 0)
    return {
      min: 0,
      max,
      normalRange: [0, warn],
      warningRange: [warn, crit],
      criticalRange: [crit, max],
    }
  }
  // Lower is worse (oil pressure, outlet pressure)
  const max = Math.max(nominal * 1.4, sensor.value ?? 0)
  return {
    min: 0,
    max,
    normalRange: [warn, max],
    warningRange: [crit, warn],
    criticalRange: [0, crit],
  }
}

const statusBadgeVariant: Record<string, 'success' | 'warning' | 'error' | 'info'> = {
  normal: 'success',
  warning: 'warning',
  critical: 'error',
  maintenance: 'info',
  offline: 'info',
}

// Scenarios act on specific sensors, so only the ones native to this
// machine type are offered (full_cascade hits every sensor).
const INJECT_SCENARIOS_BY_TYPE: Record<string, { id: string; label: string }[]> = {
  Compressor: [
    { id: 'oil_leak', label: 'Oil Leak' },
    { id: 'full_cascade', label: 'Full Cascade' },
  ],
  'Heat Exchanger': [
    { id: 'fouling_spike', label: 'Fouling Spike' },
    { id: 'full_cascade', label: 'Full Cascade' },
  ],
  Conveyor: [
    { id: 'belt_slip', label: 'Belt Slip' },
    { id: 'full_cascade', label: 'Full Cascade' },
  ],
}
const DEFAULT_INJECT_SCENARIOS = [{ id: 'full_cascade', label: 'Full Cascade' }]

const priorityVariant: Record<string, 'p1' | 'p2' | 'p3' | 'p4'> = {
  LOW: 'p1',
  MEDIUM: 'p2',
  HIGH: 'p3',
  CRITICAL: 'p4',
}

function WhatIfCard({ machineId }: { machineId: string }) {
  const { t } = useI18n()
  const [deferHours, setDeferHours] = useState(24)
  const [committedHours, setCommittedHours] = useState(24)
  const whatIfApi = useApi<WhatIfResult>(
    `/machines/${machineId}/whatif?defer_hours=${committedHours}`,
    60000,
  )
  const result = whatIfApi.data

  return (
    <Card title={t('machine.whatIf')} subtitle={t('machine.whatIfSub')}>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xs text-text-secondary shrink-0">{t('machine.deferBy')}</span>
        <input
          type="range"
          min={1}
          max={168}
          value={deferHours}
          onChange={(e) => setDeferHours(Number(e.target.value))}
          onMouseUp={() => setCommittedHours(deferHours)}
          onTouchEnd={() => setCommittedHours(deferHours)}
          className="flex-1 accent-[#448aff]"
          aria-label={t('machine.deferBy')}
        />
        <span className="text-xs font-mono text-text-primary w-16 text-right">
          {deferHours} {t('machine.hours')}
        </span>
      </div>

      {whatIfApi.error && !result ? (
        <p className="text-text-tertiary text-xs py-2">{t('machine.whatIfUnavailable')}</p>
      ) : !result ? (
        <p className="text-text-tertiary text-xs py-2">…</p>
      ) : (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2">
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">{t('machine.actNow')}</span>
            <span className="text-sm font-mono text-text-primary">
              €{result.act_now_cost_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">
              {t('machine.failureProb', { hours: result.defer_hours })}
            </span>
            <span
              className={`text-sm font-mono ${
                result.failure_probability > 0.5
                  ? 'text-alarm-p4'
                  : result.failure_probability > 0.1
                    ? 'text-alarm-p3'
                    : 'text-text-primary'
              }`}
            >
              {(result.failure_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">
              {t('machine.deferredCost')}
            </span>
            <span className="text-sm font-mono text-text-primary">
              €{result.expected_deferred_cost_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">
              {t('machine.runToFailure')}
            </span>
            <span className="text-sm font-mono text-text-secondary">
              €{result.run_to_failure_cost_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">
              {t('machine.actingNowSaves')}
            </span>
            <span
              className={`text-sm font-mono ${
                result.net_benefit_of_acting_now_eur > 0 ? 'text-success' : 'text-text-secondary'
              }`}
            >
              €{result.net_benefit_of_acting_now_eur.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-text-tertiary uppercase">{t('machine.breakeven')}</span>
            <span className="text-sm font-mono text-text-primary">
              {result.breakeven_hours != null
                ? `${result.breakeven_hours} ${t('machine.hours')}`
                : t('machine.breakevenNever')}
            </span>
          </div>
        </div>
      )}
    </Card>
  )
}

export default function MachineDetail() {
  const { id } = useParams<{ id: string }>()
  const { t } = useI18n()
  const machineApi = useApi<MachineDetailData>(`/machines/${id}`, 10000)
  const workOrdersApi = useApi<WorkOrderItem[]>(`/work-orders?machine_id=${id}`, 30000)
  const seriesApi = useApi<SensorSeries>(`/machines/${id}/sensors?minutes=60`, 15000)
  const alarmsApi = useApi<AlarmItem[]>('/alarms?active=false&limit=100', 30000)
  const [injecting, setInjecting] = useState<string | null>(null)
  const [injectMessage, setInjectMessage] = useState<string | null>(null)

  const machine = machineApi.data
  const workOrders = workOrdersApi.data ?? []
  const series = seriesApi.data?.series ?? {}
  const machineAlarms = (alarmsApi.data ?? []).filter((a) => a.machine_id === id).slice(0, 8)

  const handleInject = async (scenario: string) => {
    if (!id) return
    setInjecting(scenario)
    setInjectMessage(null)
    try {
      await apiPost(`/machines/${id}/inject-anomaly`, { scenario, ramp_seconds: 10 })
      setInjectMessage(t('machine.injected', { scenario: scenario.replace(/_/g, ' ') }))
    } catch (err) {
      setInjectMessage(
        err instanceof Error ? `${t('machine.injectionFailed')}: ${err.message}` : t('machine.injectionFailed'),
      )
    } finally {
      setInjecting(null)
    }
  }

  if (machineApi.error && !machine) {
    return (
      <div className="p-4">
        <Header title={t('machine.title')} />
        <Card>
          <p className="text-alarm-p4 text-sm">
            {t('machine.loadFailed')} {machineApi.error}
          </p>
        </Card>
      </div>
    )
  }

  if (!machine) {
    return (
      <div className="p-4">
        <Header title={t('machine.title')} />
        <Card>
          <p className="text-text-tertiary text-sm">{t('machine.loading')}</p>
        </Card>
      </div>
    )
  }

  const badgeVariant = statusBadgeVariant[machine.status]
  const statusLabel = t(`status.${machine.status}` as TranslationKey)

  // Radar: current vs nominal, both as % of critical threshold.
  const radarVariables = machine.sensors
    .filter((s) => s.critical_threshold != null && s.value != null)
    .map((s) => {
      const scale = Math.abs(s.critical_threshold ?? 1) || 1
      return {
        name: s.sensor_name.replace(/_/g, ' '),
        value: Math.min(Math.round((Math.abs(s.value ?? 0) / scale) * 100), 120),
        baseline: Math.min(Math.round((Math.abs(s.nominal_mu ?? 0) / scale) * 100), 120),
        unit: '%',
      }
    })

  return (
    <div className="p-4">
      <Header title={t('machine.title')} />

      {/* Machine Info Header */}
      <Card className="mb-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-4 flex-wrap">
            <div>
              <h2 className="text-lg font-semibold text-text-primary m-0">{machine.id}</h2>
              <p className="text-xs text-text-tertiary mt-0.5">
                {machine.name} · {machine.standard ?? machine.type}
              </p>
            </div>
            <Badge variant={badgeVariant} label={statusLabel} />
            <div className="flex flex-col">
              <span className="text-[11px] text-text-tertiary">{t('machine.rul')}</span>
              <span className="text-sm font-mono text-text-primary">
                {machine.rul_hours != null ? `${machine.rul_hours.toFixed(0)} h` : '—'}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[11px] text-text-tertiary">{t('machine.reliability')}</span>
              <span
                className={`text-sm font-mono ${
                  machine.reliability != null && machine.reliability < 50 ? 'text-alarm-p4' : 'text-text-primary'
                }`}
              >
                {machine.reliability != null ? `${machine.reliability.toFixed(1)}%` : '—'}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[11px] text-text-tertiary">{t('machine.health')}</span>
              <span className="text-sm font-mono text-text-primary">
                {machine.health_score != null ? `${(machine.health_score * 100).toFixed(0)}%` : '—'}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-text-tertiary">{t('machine.classification')}</span>
            <span className="text-sm font-mono text-text-primary">{machine.classification ?? '—'}</span>
          </div>
        </div>
      </Card>

      {/* Sensor Gauges */}
      <div className="grid grid-cols-5 gap-3 mb-3">
        <div className="col-span-2">
          <Card title={t('machine.failureModes')} subtitle={machine.failure_mode ?? undefined}>
            {machine.active_faults.length === 0 ? (
              <p className="text-text-tertiary text-xs py-4">{t('machine.noActiveFaults')}</p>
            ) : (
              machine.active_faults.map((fault, i) => (
                <div key={i} className="py-2.5 border-b border-border-subtle last:border-0">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-primary font-medium">
                      {(fault.fault_type ?? 'Unclassified anomaly').replace(/_/g, ' ')}
                    </span>
                    <div className="flex items-center gap-2">
                      <Badge variant={fault.severity === 'CRITICAL' ? 'error' : 'warning'} label={fault.severity} />
                      {fault.confidence != null && (
                        <span className="text-xs font-mono text-text-secondary">
                          {(fault.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-[10px] text-text-tertiary mt-1">
                    {t('machine.topSensor')} {fault.top_contributing_sensor ?? '—'} · score{' '}
                    {fault.anomaly_score?.toFixed(2) ?? '—'} ·{' '}
                    {new Date(fault.detected_at).toLocaleString()}
                  </p>
                </div>
              ))
            )}
          </Card>
        </div>
        <div className="col-span-3">
          <Card title={t('machine.sensorCluster')} subtitle={t('machine.sensorClusterSub')}>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              {machine.sensors.map((s) => {
                if (s.value == null) {
                  return (
                    <div key={s.sensor_name} className="text-xs text-text-tertiary py-2">
                      {s.sensor_name.replace(/_/g, ' ')}: {t('machine.noData')}
                    </div>
                  )
                }
                const ranges = gaugeRanges(s)
                return (
                  <Gauge
                    key={s.sensor_name}
                    label={s.sensor_name.replace(/_/g, ' ')}
                    value={s.value}
                    min={ranges.min}
                    max={ranges.max}
                    unit={s.unit ?? ''}
                    normalRange={ranges.normalRange}
                    warningRange={ranges.warningRange}
                    criticalRange={ranges.criticalRange}
                    dataPoints={s.history}
                  />
                )
              })}
            </div>
          </Card>
        </div>
      </div>

      {/* Radar + What-If + Work Orders */}
      <div className="grid grid-cols-5 gap-3">
        <div className="col-span-2">
          <Card title={t('machine.sensorProfile')} subtitle={t('machine.sensorProfileSub')}>
            {radarVariables.length >= 3 ? (
              <div className="flex justify-center">
                <RadarPlot variables={radarVariables} size={220} />
              </div>
            ) : (
              <p className="text-text-tertiary text-xs py-4">{t('machine.notEnoughSensors')}</p>
            )}
          </Card>
        </div>
        <div className="col-span-3 flex flex-col gap-3">
          {id && <WhatIfCard machineId={id} />}
          <Card title={t('machine.workOrders')}>
            {workOrders.length === 0 ? (
              <p className="text-text-tertiary text-xs py-4">{t('machine.noWorkOrders')}</p>
            ) : (
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-text-tertiary border-b border-border-default">
                    <th className="pb-2 pr-4 font-medium">ID</th>
                    <th className="pb-2 pr-4 font-medium">{t('machine.col.action')}</th>
                    <th className="pb-2 pr-4 font-medium">{t('machine.col.status')}</th>
                    <th className="pb-2 pr-4 font-medium">{t('machine.col.priority')}</th>
                    <th className="pb-2 font-medium">{t('machine.col.estCost')}</th>
                  </tr>
                </thead>
                <tbody>
                  {workOrders.map((wo) => (
                    <tr key={wo.id} className="border-b border-border-subtle last:border-0">
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {wo.work_order_number ?? wo.id.slice(0, 8)}
                      </td>
                      <td className="py-2 pr-4 text-text-primary">
                        {wo.recommended_action ?? wo.fault_type ?? '—'}
                      </td>
                      <td className="py-2 pr-4">
                        <span
                          className={`text-[11px] capitalize ${
                            wo.status === 'PENDING'
                              ? 'text-alarm-p3'
                              : wo.status === 'IN_PROGRESS'
                                ? 'text-alarm-p0'
                                : 'text-success'
                          }`}
                        >
                          {wo.status.replace('_', ' ').toLowerCase()}
                        </span>
                      </td>
                      <td className="py-2 pr-4">
                        <Badge variant={priorityVariant[wo.priority] ?? 'p2'} label={wo.priority} />
                      </td>
                      <td className="py-2 font-mono text-text-secondary">
                        {wo.estimated_cost_eur != null ? `€${wo.estimated_cost_eur.toLocaleString()}` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      </div>

      {/* Sensor Trends */}
      <Card title={t('machine.sensorTrends')} subtitle={t('machine.sensorTrendsSub')} className="mt-3">
        {Object.keys(series).length === 0 ? (
          <p className="text-text-tertiary text-xs py-4">{t('machine.noReadings')}</p>
        ) : (
          <div className="grid grid-cols-2 gap-x-6 gap-y-4">
            {machine.sensors
              .filter((s) => (series[s.sensor_name] ?? []).length > 1)
              .map((s) => (
                <div key={s.sensor_name}>
                  <p className="text-xs text-text-primary font-medium mb-1">
                    {s.sensor_name.replace(/_/g, ' ')}
                    {s.unit ? <span className="text-text-tertiary"> ({s.unit})</span> : null}
                  </p>
                  <SensorTrendChart
                    sensorName={s.sensor_name}
                    unit={s.unit}
                    points={series[s.sensor_name]}
                    warningThreshold={s.warning_threshold}
                    criticalThreshold={s.critical_threshold}
                  />
                </div>
              ))}
          </div>
        )}
      </Card>

      {/* Alarm History + Demo Controls */}
      <div className="grid grid-cols-5 gap-3 mt-3">
        <div className="col-span-3">
          <Card title={t('machine.alarmHistory')} subtitle={t('machine.alarmHistorySub')}>
            {machineAlarms.length === 0 ? (
              <p className="text-text-tertiary text-xs py-4">{t('machine.noAlarms')}</p>
            ) : (
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-text-tertiary border-b border-border-default">
                    <th className="pb-2 pr-3 font-medium">{t('machine.col.when')}</th>
                    <th className="pb-2 pr-3 font-medium">{t('machine.col.fault')}</th>
                    <th className="pb-2 pr-3 font-medium">{t('machine.col.severity')}</th>
                    <th className="pb-2 pr-3 font-medium">{t('machine.col.state')}</th>
                    <th className="pb-2 font-medium">{t('machine.col.score')}</th>
                  </tr>
                </thead>
                <tbody>
                  {machineAlarms.map((a) => (
                    <tr key={a.id} className="border-b border-border-subtle last:border-0">
                      <td className="py-2 pr-3 font-mono text-text-secondary whitespace-nowrap">
                        {new Date(a.created_at).toLocaleString()}
                      </td>
                      <td className="py-2 pr-3 text-text-primary">
                        {(a.fault_type ?? a.top_contributing_sensor ?? 'Anomaly').replace(/_/g, ' ')}
                      </td>
                      <td className="py-2 pr-3">
                        <Badge variant={a.severity === 'CRITICAL' ? 'error' : 'warning'} label={a.severity} />
                      </td>
                      <td className="py-2 pr-3 text-text-secondary capitalize">
                        {a.status.replace(/_/g, ' ').toLowerCase()}
                      </td>
                      <td className="py-2 font-mono text-text-secondary">
                        {a.anomaly_score?.toFixed(2) ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
        <div className="col-span-2">
          <Card title={t('machine.demoControls')} subtitle={t('machine.demoControlsSub')}>
            <div className="grid grid-cols-2 gap-2">
              {(INJECT_SCENARIOS_BY_TYPE[machine.type] ?? DEFAULT_INJECT_SCENARIOS).map((sc) => (
                <Button
                  key={sc.id}
                  variant="secondary"
                  size="sm"
                  onClick={() => handleInject(sc.id)}
                  disabled={injecting !== null}
                >
                  <span aria-hidden="true">⚡</span>
                  {injecting === sc.id ? t('machine.injecting') : sc.label}
                </Button>
              ))}
            </div>
            {injectMessage && (
              <p className="text-[11px] text-text-secondary mt-2 leading-relaxed">{injectMessage}</p>
            )}
            <p className="text-[10px] text-text-tertiary mt-2 leading-relaxed">
              {t('machine.injectionNote')}
            </p>
          </Card>
        </div>
      </div>
    </div>
  )
}
