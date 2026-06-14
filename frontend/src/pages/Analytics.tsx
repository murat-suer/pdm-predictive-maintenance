import Card from '../components/Card'
import Header from '../components/Header'
import { useApi } from '../api/hooks'
import { useI18n } from '../i18n'
import type {
  DecisionMixResponse,
  DecisionStatsResponse,
  FaultDistributionResponse,
  MaintenanceTimelineResponse,
  MhiHistoryResponse,
  SavingsTimeseriesResponse,
} from '../api/types'
import DonutChart from '../components/charts/DonutChart'
import BarChart from '../components/charts/BarChart'
import AreaChart from '../components/charts/AreaChart'
import MultiLineChart from '../components/charts/MultiLineChart'
import type { LineSeriesData } from '../components/charts/MultiLineChart'
import { PALETTE } from '../components/charts/palette'

// ISA-101 semantic colours per decision scenario
const SCENARIO_COLOR: Record<string, string> = {
  OBSERVE: PALETTE.OBSERVE,
  PLANNED: PALETTE.PLANNED,
  DISPATCH_TECHNICIAN: PALETTE.DISPATCH_TECHNICIAN,
  REDUCE_LOAD: PALETTE.REDUCE_LOAD,
  SHUTDOWN: PALETTE.SHUTDOWN,
}

function fmtEur(v: number): string {
  return `€${Math.max(v, 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`
}

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function fmtLabel(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// ── KPI card ──────────────────────────────────────────────────────────────────

interface KpiCardProps {
  label: string
  value: string | null
  color?: string
  sub?: string
}

function KpiCard({ label, value, color = 'text-text-primary', sub }: KpiCardProps) {
  return (
    <Card>
      <div className="flex flex-col gap-0.5">
        <span className="text-text-tertiary text-[11px] uppercase tracking-wider">{label}</span>
        <span className={`${color} text-2xl font-semibold font-mono tabular-nums`}>
          {value ?? '—'}
        </span>
        {sub && <span className="text-text-tertiary text-[10px]">{sub}</span>}
      </div>
    </Card>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Analytics() {
  const { t } = useI18n()

  const decisionMix = useApi<DecisionMixResponse>('/analytics/decision-mix', 60000)
  const faultDist = useApi<FaultDistributionResponse>('/analytics/fault-distribution', 60000)
  const savingsTs = useApi<SavingsTimeseriesResponse>(
    '/analytics/savings-timeseries?buckets=12',
    60000,
  )
  const mhiHistory = useApi<MhiHistoryResponse>(
    '/analytics/mhi-history?buckets=24',
    60000,
  )
  const timeline = useApi<MaintenanceTimelineResponse>('/analytics/maintenance-timeline', 60000)
  const stats = useApi<DecisionStatsResponse>('/analytics/decision-stats', 60000)

  const loading =
    decisionMix.loading ||
    faultDist.loading ||
    savingsTs.loading ||
    mhiHistory.loading ||
    timeline.loading ||
    stats.loading

  const anyError =
    decisionMix.error ||
    faultDist.error ||
    savingsTs.error ||
    mhiHistory.error ||
    timeline.error ||
    stats.error

  // ── Decision mix donut ──────────────────────────────────────────────────────
  const donutSlices = (decisionMix.data?.items ?? []).map((item) => ({
    label: item.scenario,
    value: item.count,
    color: SCENARIO_COLOR[item.scenario] ?? PALETTE.NEUTRAL,
  }))

  // ── Fault distribution horizontal bar ──────────────────────────────────────
  const faultItems = [...(faultDist.data?.items ?? [])].sort((a, b) => b.count - a.count)
  const faultLabels = faultItems.map((f) => f.fault_type)
  const faultValues = faultItems.map((f) => f.count)
  const unclassifiedIdx = faultLabels.findIndex((l) =>
    l.toUpperCase().includes('UNCLASSIFIED'),
  )
  const faultColors = faultLabels.map((_, i) =>
    i === unclassifiedIdx ? PALETTE.ALARM_P3 : PALETTE.NEUTRAL,
  )

  // ── Cumulative savings area chart ──────────────────────────────────────────
  const savingsPoints = savingsTs.data?.points ?? []
  const savingsLabels = savingsPoints.map((p) => fmtLabel(p.t))
  const savingsValues = savingsPoints.map((p) => p.cumulative_eur)

  // ── MHI multi-line chart ───────────────────────────────────────────────────
  const mhiMachines = mhiHistory.data?.machines ?? []
  // Union of all timestamps across machines (sorted)
  const mhiAllTs = Array.from(
    new Set(mhiMachines.flatMap((m) => m.points.map((p) => p.t))),
  ).sort()
  const mhiLabels = mhiAllTs.map((t) => fmtLabel(t))
  const mhiSeries: LineSeriesData[] = mhiMachines.map((machine, idx) => {
    const byTime = new Map(machine.points.map((p) => [p.t, p.mhi]))
    return {
      id: machine.machine_id,
      label: machine.machine_id,
      values: mhiAllTs.map((t) => byTime.get(t) ?? NaN),
      color: PALETTE.MACHINE_LINES[idx % PALETTE.MACHINE_LINES.length],
    }
  })

  // ── Bot vs human bar ───────────────────────────────────────────────────────
  const bvh = stats.data?.bot_vs_human
  const bvhLabels = ['Bot', 'Human']
  const bvhValues = [bvh?.bot ?? 0, bvh?.human ?? 0]
  const bvhColors = [PALETTE.NEUTRAL, PALETTE.PLANNED]

  // ── Timeline events ────────────────────────────────────────────────────────
  const events = [...(timeline.data?.events ?? [])].sort(
    (a, b) => new Date(b.performed_at).getTime() - new Date(a.performed_at).getTime(),
  )

  // ── KPI values ─────────────────────────────────────────────────────────────
  const netSaved =
    savingsPoints.length > 0
      ? fmtEur(savingsPoints[savingsPoints.length - 1].cumulative_eur)
      : null

  const avgResponseSec = stats.data ? `${stats.data.avg_response_time_s.toFixed(0)}s` : null
  const overrideRateVal = stats.data ? fmtPct(stats.data.override_rate) : null
  const autoApprovedVal = stats.data
    ? `${stats.data.auto_approved}/${stats.data.total}`
    : null
  const bvhLabel =
    bvh != null ? `${bvh.bot} / ${bvh.human}` : null

  const windowLabel = decisionMix.data?.window_started_at
    ? `${t('analytics.window')} · since ${fmtTime(decisionMix.data.window_started_at)}`
    : t('analytics.window')

  return (
    <div className="p-4">
      <Header title={t('analytics.title')} />

      {/* Loading / error banners */}
      {loading && (
        <Card className="mb-4">
          <p className="text-text-tertiary text-xs">{t('analytics.loading')}</p>
        </Card>
      )}
      {!loading && anyError && (
        <Card className="mb-4">
          <p className="text-alarm-p4 text-xs">{t('analytics.error')}</p>
        </Card>
      )}

      {/* Window label */}
      <p className="text-text-tertiary text-xs mb-4 font-mono">{windowLabel}</p>

      {/* ── KPI row ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 mb-4">
        <KpiCard
          label={t('analytics.kpi.netSaved')}
          value={netSaved}
          color="text-success"
          sub={`${events.length} ${t('analytics.kpi.events').toLowerCase()}`}
        />
        <KpiCard
          label={t('analytics.kpi.decisions')}
          value={stats.data ? String(stats.data.total) : null}
        />
        <KpiCard
          label={t('analytics.kpi.events')}
          value={events.length > 0 ? String(events.length) : null}
        />
        <KpiCard
          label={t('analytics.kpi.reliability')}
          value={
            mhiMachines.length > 0
              ? (() => {
                  const lastMhis = mhiMachines
                    .map((m) => m.points[m.points.length - 1]?.mhi)
                    .filter((v) => v != null) as number[]
                  const avg = lastMhis.reduce((s, v) => s + v, 0) / (lastMhis.length || 1)
                  return `${avg.toFixed(0)}`
                })()
              : null
          }
          color={
            mhiMachines.length > 0
              ? (() => {
                  const lastMhis = mhiMachines
                    .map((m) => m.points[m.points.length - 1]?.mhi)
                    .filter((v) => v != null) as number[]
                  const avg = lastMhis.reduce((s, v) => s + v, 0) / (lastMhis.length || 1)
                  return avg < 55 ? 'text-alarm-p4' : avg < 70 ? 'text-alarm-p3' : 'text-text-primary'
                })()
              : 'text-text-primary'
          }
        />
        <KpiCard
          label={t('analytics.kpi.overrideRate')}
          value={overrideRateVal}
          color={
            stats.data
              ? stats.data.override_rate > 0.3
                ? 'text-alarm-p3'
                : 'text-text-primary'
              : 'text-text-primary'
          }
        />
        <KpiCard
          label={t('analytics.kpi.autoApproved')}
          value={autoApprovedVal}
        />
        <KpiCard
          label={t('analytics.kpi.avgResponse')}
          value={avgResponseSec}
          sub={t('analytics.kpi.botVsHuman') + ': ' + (bvhLabel ?? '—')}
        />
      </div>

      {/* ── Row: Decision mix + Fault distribution ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <Card title={t('analytics.decisionMix')} subtitle={t('analytics.decisionMixSub')}>
          <DonutChart
            slices={donutSlices}
            ariaLabel={t('analytics.decisionMix')}
            height={180}
          />
        </Card>

        <Card title={t('analytics.faultDist')} subtitle={t('analytics.faultDistSub')}>
          <BarChart
            labels={faultLabels}
            values={faultValues}
            colors={faultColors}
            ariaLabel={t('analytics.faultDist')}
            horizontal
            height={Math.max(120, faultLabels.length * 28)}
            highlightIndex={unclassifiedIdx >= 0 ? unclassifiedIdx : undefined}
          />
        </Card>
      </div>

      {/* ── Cumulative savings area chart ── */}
      <Card
        title={t('analytics.savings')}
        subtitle={t('analytics.savingsSub')}
        className="mb-4"
      >
        <AreaChart
          labels={savingsLabels}
          values={savingsValues}
          color={PALETTE.GOOD}
          ariaLabel={t('analytics.savings')}
          height={180}
          formatValue={(v) => `€${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
        />
      </Card>

      {/* ── MHI multi-line trend ── */}
      <Card
        title={t('analytics.mhi')}
        subtitle={t('analytics.mhiSub')}
        className="mb-4"
      >
        <MultiLineChart
          labels={mhiLabels}
          series={mhiSeries}
          ariaLabel={t('analytics.mhi')}
          height={220}
          yMin={0}
          yMax={100}
          formatValue={(v) => v.toFixed(0)}
        />
      </Card>

      {/* ── Row: Maintenance timeline + Bot-vs-human ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 mb-4">
        {/* Timeline — takes 2/3 of the row */}
        <Card
          title={t('analytics.timeline')}
          subtitle={t('analytics.timelineSub')}
          className="lg:col-span-2"
        >
          {events.length === 0 ? (
            <p className="text-text-tertiary text-xs py-4 text-center">
              {t('analytics.noEvents')}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-text-tertiary border-b border-border-default">
                    <th className="pb-2 pr-3 font-medium">{t('analytics.col.machine')}</th>
                    <th className="pb-2 pr-3 font-medium">{t('analytics.col.scenario')}</th>
                    <th className="pb-2 pr-3 font-medium">{t('analytics.col.when')}</th>
                    <th className="pb-2 pr-3 font-medium text-right">{t('analytics.col.cost')}</th>
                    <th className="pb-2 pr-3 font-medium text-right">
                      {t('analytics.col.savings')}
                    </th>
                    <th className="pb-2 font-medium text-right">{t('analytics.col.downtime')}</th>
                  </tr>
                </thead>
                <tbody>
                  {events.slice(0, 15).map((ev, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-border-subtle last:border-0"
                    >
                      <td className="py-2 pr-3 font-mono text-alarm-p0">{ev.machine_id}</td>
                      <td className="py-2 pr-3">
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                          style={{
                            color: SCENARIO_COLOR[ev.scenario] ?? PALETTE.NEUTRAL,
                            backgroundColor: `${SCENARIO_COLOR[ev.scenario] ?? PALETTE.NEUTRAL}18`,
                          }}
                        >
                          {ev.scenario}
                        </span>
                      </td>
                      <td className="py-2 pr-3 text-text-secondary font-mono">
                        {fmtTime(ev.performed_at)}
                      </td>
                      <td className="py-2 pr-3 font-mono text-right text-text-secondary">
                        {fmtEur(ev.actual_cost_eur)}
                      </td>
                      <td className="py-2 pr-3 font-mono text-right text-success">
                        {ev.savings_eur > 0 ? fmtEur(ev.savings_eur) : '—'}
                      </td>
                      <td className="py-2 font-mono text-right text-text-secondary">
                        {ev.downtime_minutes > 0 ? `${ev.downtime_minutes.toFixed(0)}m` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* Bot vs Human + avg response */}
        <Card title={t('analytics.botVsHuman')} subtitle={t('analytics.botVsHumanSub')}>
          <BarChart
            labels={bvhLabels}
            values={bvhValues}
            colors={bvhColors}
            ariaLabel={t('analytics.botVsHuman')}
            height={140}
          />
          {stats.data && (
            <div className="mt-3 space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-text-tertiary">{t('analytics.kpi.overrideRate')}</span>
                <span className="font-mono text-text-primary">
                  {fmtPct(stats.data.override_rate)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-tertiary">{t('analytics.kpi.avgResponse')}</span>
                <span className="font-mono text-text-primary">
                  {stats.data.avg_response_time_s.toFixed(0)}s
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-tertiary">{t('analytics.kpi.autoApproved')}</span>
                <span className="font-mono text-text-primary">
                  {stats.data.auto_approved} / {stats.data.total}
                </span>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
