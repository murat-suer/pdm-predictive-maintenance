import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Header from '../components/Header'
import { apiPost } from '../api/client'
import { useApi } from '../api/hooks'
import { useI18n, type TranslationKey } from '../i18n'
import type { DecisionResolveResult, PendingDecision } from '../api/types'

const SCENARIO_ROLE: Record<string, string> = {
  OBSERVE: 'Any',
  DISPATCH_TECHNICIAN: 'Operator',
  REDUCE_LOAD: 'Operator',
  PLANNED: 'Supervisor',
  SHUTDOWN: 'Manager',
}

function WatchdogRing({ remainingSeconds, totalSeconds }: { remainingSeconds: number; totalSeconds: number }) {
  const { t } = useI18n()
  const r = 40
  const circ = 2 * Math.PI * r
  const pct = totalSeconds > 0 ? Math.max(remainingSeconds, 0) / totalSeconds : 0
  const dashOffset = circ * (1 - pct)
  const color = pct < 0.2 ? '#ff1744' : pct < 0.5 ? '#ffd600' : '#448aff'
  const minutes = Math.floor(Math.max(remainingSeconds, 0) / 60)
  const seconds = Math.max(remainingSeconds, 0) % 60

  return (
    <div className="flex items-center gap-3">
      <svg width="100" height="100" viewBox="0 0 100 100" className="shrink-0">
        <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
        <circle
          cx="50"
          cy="50"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circ}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform="rotate(-90, 50, 50)"
        />
        <text x="50" y="46" textAnchor="middle" fill="#f7f8f8" fontSize="13" fontWeight="600" fontFamily="'IBM Plex Mono', monospace">
          {minutes}:{seconds.toString().padStart(2, '0')}
        </text>
        <text x="50" y="60" textAnchor="middle" fill="#6b6b73" fontSize="9" fontFamily="'IBM Plex Sans', sans-serif">
          min:sec
        </text>
      </svg>
      <div className="flex flex-col">
        <span className="text-xs text-text-primary font-medium">{t('decision.autoApproveTimer')}</span>
        <span className="text-[11px] text-text-tertiary">{t('decision.autoApproveAtZero')}</span>
      </div>
    </div>
  )
}

function SensorBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0
  return (
    <div className="flex items-center gap-2">
      <span className="w-36 text-xs text-text-secondary truncate shrink-0">{label}</span>
      <div className="flex-1 h-3 bg-bg-secondary rounded-sm overflow-hidden border border-border-subtle">
        <div
          className="h-full rounded-sm transition-all duration-500"
          style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color, opacity: 0.8 }}
        />
      </div>
      <span className="w-12 text-xs text-right font-mono text-text-secondary">{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

function useCountdown(dueAt: string | null): number {
  const [remaining, setRemaining] = useState(0)
  useEffect(() => {
    if (!dueAt) return
    const update = () => setRemaining(Math.floor((new Date(dueAt).getTime() - Date.now()) / 1000))
    update()
    const interval = setInterval(update, 1000)
    return () => clearInterval(interval)
  }, [dueAt])
  return remaining
}

export default function DecisionScreen() {
  const { t } = useI18n()
  const pendingApi = useApi<PendingDecision[]>('/decisions/pending', 10000)
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<DecisionResolveResult | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const decision = pendingApi.data?.[0] ?? null
  const remainingSeconds = useCountdown(decision?.due_at ?? null)

  // SHAP values → top sensor contribution bars (normalized to the max |value|).
  const shapEntries = decision?.shap_values
    ? Object.entries(decision.shap_values)
        .map(([key, value]) => ({ label: key.replace(/_value$/, '').replace(/_/g, ' '), value: Math.abs(value) }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 6)
    : []
  const shapMax = shapEntries[0]?.value ?? 1
  const palette = ['#ff1744', '#ff8c00', '#ffd600', '#448aff', '#448aff', '#448aff']
  const sensorContributions = shapEntries.map((e, i) => ({
    ...e,
    normalized: shapMax > 0 ? e.value / shapMax : 0,
    color: palette[i],
  }))

  const recommended = decision?.scenarios.find((s) => s.is_recommended) ?? null
  const totalSeconds =
    decision?.due_at && decision.created_at
      ? Math.max(
          Math.floor((new Date(decision.due_at).getTime() - new Date(decision.created_at).getTime()) / 1000),
          1,
        )
      : 180

  const handleExecute = async () => {
    if (!decision || !selectedScenario) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const response = await apiPost<DecisionResolveResult>(`/decisions/${decision.id}/resolve`, {
        scenario_id: selectedScenario,
        operator_role: 'SUPERVISOR',
        operator_id: 'HUMAN-OP-1',
      })
      setResult(response)
      setSelectedScenario(null)
      pendingApi.refetch()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Resolution failed')
    } finally {
      setSubmitting(false)
    }
  }

  if (pendingApi.error && !decision) {
    return (
      <div className="p-4">
        <Header title={t('decision.title')} />
        <Card>
          <p className="text-alarm-p4 text-sm">
            {t('decision.backendUnreachable')} {pendingApi.error}
          </p>
        </Card>
      </div>
    )
  }

  if (!decision) {
    return (
      <div className="p-4">
        <Header title={t('decision.title')} />
        {result && (
          <Card className="mb-3 border-success/50">
            <p className="text-success text-sm font-medium">
              ✓ {t('decision.executed')} {result.chosen_scenario_id}
              {result.overridden && ` ${t('decision.overriddenNote')}`}
              {result.work_order_id && ` · ${t('decision.workOrder')} ${result.work_order_id}`}
            </p>
          </Card>
        )}
        <Card>
          <p className="text-text-secondary text-sm">{t('decision.noPending')}</p>
          <p className="text-text-tertiary text-xs mt-1">{t('decision.noPendingHint')}</p>
        </Card>
      </div>
    )
  }

  const scenarioGridClass =
    decision.scenarios.length >= 5 ? 'grid grid-cols-5 gap-3' : 'grid grid-cols-4 gap-3'

  return (
    <div className="p-4">
      <Header title={t('decision.title')} />

      {/* Human-oversight transparency note (EU AI Act) */}
      <Card className="mb-3">
        <p className="text-[11px] text-text-tertiary leading-relaxed">
          <span className="text-text-secondary font-medium">{t('decision.oversightTitle')}</span>{' '}
          {t('decision.oversightBody')}
        </p>
      </Card>

      {/* Active Alarm Banner */}
      <Card className="mb-3 border-alarm-p4/50">
        <div className="flex items-center gap-3 min-h-14">
          <div className="w-10 h-10 rounded-full bg-alarm-p4/20 border border-alarm-p4/40 flex items-center justify-center shrink-0">
            <span className="text-alarm-p4 font-bold text-lg">!</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-alarm-p4 text-sm font-semibold">
                {decision.severity ?? 'ALARM'} — {(decision.fault_type ?? 'Anomaly detected').replace(/_/g, ' ')}
              </span>
              <Badge variant="p4" label={t('decision.pending')} />
            </div>
            <p className="text-text-secondary text-xs mt-0.5">
              {t('decision.machine')}{' '}
              <Link
                to={`/machines/${decision.machine_id}`}
                className="text-alarm-p0 font-medium hover:underline"
              >
                {decision.machine_id} ↗
              </Link>
              {decision.rul_hours != null && ` · RUL: ${decision.rul_hours.toFixed(0)} h`}
              {decision.anomaly_score != null && ` · Anomaly score: ${decision.anomaly_score.toFixed(2)}`}
            </p>
          </div>
          {decision.ai_recommendation && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-[11px] text-text-tertiary">{t('decision.aiRecommends')}</span>
              <Badge variant="p3" label={decision.ai_recommendation.replace(/_/g, ' ')} />
            </div>
          )}
        </div>
      </Card>

      {submitError && (
        <Card className="mb-3 border-alarm-p4/50">
          <p className="text-alarm-p4 text-xs">
            {t('decision.executionFailed')} {submitError}
          </p>
        </Card>
      )}

      <div className="grid grid-cols-5 gap-3">
        {/* Sensor Contribution Bars */}
        <div className="col-span-3">
          <Card title={t('decision.sensorContribution')} subtitle={t('decision.sensorContributionSub')}>
            {sensorContributions.length === 0 ? (
              <p className="text-text-tertiary text-xs py-4">{t('decision.noShap')}</p>
            ) : (
              <div className="flex flex-col gap-2">
                {sensorContributions.map((s) => (
                  <SensorBar key={s.label} label={s.label} value={s.normalized} max={1} color={s.color} />
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Watchdog Timer */}
        <div className="col-span-2">
          <Card title={t('decision.responseTimer')}>
            <div className="flex items-center justify-center">
              <WatchdogRing remainingSeconds={remainingSeconds} totalSeconds={totalSeconds} />
            </div>
            <p className="text-[11px] text-text-tertiary text-center mt-2">
              {t('decision.pendingSince')} {new Date(decision.created_at).toLocaleTimeString()}
            </p>
          </Card>
        </div>
      </div>

      {/* Decision Options */}
      <Card title={t('decision.options')} subtitle={t('decision.optionsSub')} className="mt-3">
        <div className={scenarioGridClass}>
          {decision.scenarios.map((opt) => {
            const descriptionKey = `scenario.${opt.scenario}` as TranslationKey
            const isRecommended = opt.scenario === recommended?.scenario
            const isSelected = selectedScenario === opt.scenario

            return (
              <button
                key={opt.scenario}
                onClick={() => setSelectedScenario(opt.scenario)}
                className={`flex flex-col gap-2 p-4 rounded-lg border transition-all duration-150 text-left min-h-14 ${
                  isSelected
                    ? 'border-alarm-p0 bg-alarm-p0/10'
                    : isRecommended
                      ? 'border-alarm-p3/50 bg-alarm-p3/5'
                      : 'border-border-default bg-bg-elevated hover:bg-bg-hover'
                } ${opt.scenario === 'SHUTDOWN' ? 'border-alarm-p4/30 hover:border-alarm-p4/60' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <span
                    className={`text-sm font-semibold ${
                      opt.scenario === 'SHUTDOWN' ? 'text-alarm-p4' : 'text-text-primary'
                    }`}
                  >
                    {opt.scenario.replace(/_/g, ' ')}
                  </span>
                  {isRecommended && <Badge variant="warning" label={t('decision.recommended')} />}
                </div>

                <p className="text-[11px] text-text-secondary leading-tight">{t(descriptionKey)}</p>

                <div className="grid grid-cols-2 gap-2 mt-1">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-text-tertiary uppercase">{t('decision.directCost')}</span>
                    <span className="text-xs font-mono text-text-primary">
                      €{opt.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] text-text-tertiary uppercase" title={t('decision.riskAdjTooltip')}>
                      {t('decision.riskAdj')}
                    </span>
                    <span className="text-xs font-mono text-text-secondary">
                      €{(opt.expected_cost ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-1 mt-1">
                  <span className="text-[10px] text-text-tertiary">🔒</span>
                  <span className="text-[10px] text-text-tertiary">{SCENARIO_ROLE[opt.scenario] ?? 'Any'}</span>
                </div>
              </button>
            )
          })}
        </div>
        {decision.scenarios.length === 0 && (
          <p className="text-text-tertiary text-xs py-4">{t('decision.scenariosPending')}</p>
        )}
      </Card>

      {/* Execute Button */}
      {selectedScenario && (
        <div className="flex justify-end mt-4">
          <Button
            variant={selectedScenario === 'SHUTDOWN' ? 'alarm' : 'primary'}
            size="lg"
            className="min-h-14 px-8 font-semibold"
            onClick={handleExecute}
            disabled={submitting}
          >
            {submitting ? t('decision.executing') : `${t('decision.execute')} ${selectedScenario.replace(/_/g, ' ')}`}
          </Button>
        </div>
      )}
    </div>
  )
}
