import Card from '../components/Card'
import Header from '../components/Header'
import { useApi } from '../api/hooks'
import { useI18n, type TranslationKey } from '../i18n'
import type { FleetSummary, PendingDecision, SavingsSummary } from '../api/types'

const PIPELINE_STEPS: { icon: string; labelKey: TranslationKey; descKey: TranslationKey }[] = [
  { icon: '📡', labelKey: 'system.step.sense', descKey: 'system.step.senseDesc' },
  { icon: '🧠', labelKey: 'system.step.detect', descKey: 'system.step.detectDesc' },
  { icon: '📈', labelKey: 'system.step.predict', descKey: 'system.step.predictDesc' },
  { icon: '✅', labelKey: 'system.step.decide', descKey: 'system.step.decideDesc' },
  { icon: '⚙', labelKey: 'system.step.act', descKey: 'system.step.actDesc' },
  { icon: '💶', labelKey: 'system.step.account', descKey: 'system.step.accountDesc' },
]

const DESIGN_DECISIONS: { qKey: TranslationKey; aKey: TranslationKey; open?: boolean }[] = [
  { qKey: 'system.q.oversight', aKey: 'system.a.oversight', open: true },
  { qKey: 'system.q.expectedCost', aKey: 'system.a.expectedCost' },
  { qKey: 'system.q.physics', aKey: 'system.a.physics' },
  { qKey: 'system.q.xgboost', aKey: 'system.a.xgboost' },
  { qKey: 'system.q.redis', aKey: 'system.a.redis' },
  { qKey: 'system.q.timescale', aKey: 'system.a.timescale' },
  { qKey: 'system.q.mhi', aKey: 'system.a.mhi' },
]

const ASSUMPTION_ROWS: { c: TranslationKey; d: TranslationKey; p: TranslationKey }[] = [
  { c: 'system.row.weibull', d: 'system.row.weibullDemo', p: 'system.row.weibullProd' },
  { c: 'system.row.sensors', d: 'system.row.sensorsDemo', p: 'system.row.sensorsProd' },
  { c: 'system.row.speed', d: 'system.row.speedDemo', p: 'system.row.speedProd' },
  { c: 'system.row.operator', d: 'system.row.operatorDemo', p: 'system.row.operatorProd' },
  { c: 'system.row.financials', d: 'system.row.financialsDemo', p: 'system.row.financialsProd' },
  { c: 'system.row.alarm', d: 'system.row.alarmDemo', p: 'system.row.alarmProd' },
]

export default function SystemOverview() {
  const { t } = useI18n()
  const summaryApi = useApi<FleetSummary>('/fleet/summary', 30000)
  const savingsApi = useApi<SavingsSummary>('/savings', 60000)
  const pendingApi = useApi<PendingDecision[]>('/decisions/pending', 30000)
  const summary = summaryApi.data

  return (
    <div className="p-4">
      <Header title={t('system.title')} />

      {/* Hero */}
      <Card className="mb-3">
        <div className="py-4 text-center">
          <h1 className="text-2xl font-semibold text-text-primary m-0 tracking-tight">
            PDM Intelligence
          </h1>
          <p className="text-sm text-text-secondary mt-2 max-w-2xl mx-auto leading-relaxed">
            {t('system.heroSubtitle')}
          </p>
          <div className="flex items-center justify-center gap-3 mt-3">
            <span className="text-[11px] text-success border border-success/40 bg-success/10 px-2 py-0.5 rounded">
              {t('system.liveDemo')}
            </span>
            <span className="text-[11px] text-text-tertiary border border-border-subtle px-2 py-0.5 rounded font-mono">
              v3.0.0
            </span>
          </div>
        </div>
      </Card>

      {/* Problem / Solution */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <Card className="border-alarm-p4/30">
          <h3 className="text-sm font-semibold text-alarm-p4 mt-0 mb-2">{t('system.reactiveTitle')}</h3>
          <ul className="text-xs text-text-secondary leading-relaxed pl-4 m-0 flex flex-col gap-1.5">
            <li>{t('system.reactive1')}</li>
            <li>{t('system.reactive2')}</li>
            <li>{t('system.reactive3')}</li>
            <li>{t('system.reactive4')}</li>
          </ul>
        </Card>
        <Card className="border-success/30">
          <h3 className="text-sm font-semibold text-success mt-0 mb-2">{t('system.pdmTitle')}</h3>
          <ul className="text-xs text-text-secondary leading-relaxed pl-4 m-0 flex flex-col gap-1.5">
            <li>{t('system.pdm1')}</li>
            <li>{t('system.pdm2')}</li>
            <li>{t('system.pdm3')}</li>
            <li>{t('system.pdm4')}</li>
          </ul>
        </Card>
      </div>

      {/* The closed loop */}
      <Card title={t('system.howItWorks')} className="mb-3">
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
          {PIPELINE_STEPS.map((step, i) => (
            <div
              key={step.labelKey}
              className="flex flex-col items-center text-center p-3 rounded-lg border border-border-subtle bg-bg-secondary relative"
            >
              <span className="text-xl" aria-hidden="true">{step.icon}</span>
              <span className="text-[11px] font-semibold text-text-primary mt-1.5 tracking-wider">
                {t(step.labelKey)}
              </span>
              <span className="text-[10px] text-text-tertiary mt-1 leading-relaxed">
                {t(step.descKey)}
              </span>
              {i < PIPELINE_STEPS.length - 1 && (
                <span className="hidden xl:block absolute -right-2.5 top-1/2 -translate-y-1/2 text-text-tertiary" aria-hidden="true">
                  →
                </span>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Design decisions */}
      <Card title={t('system.designDecisions')} className="mb-3">
        <div className="flex flex-col gap-1">
          {DESIGN_DECISIONS.map((d) => (
            <details
              key={d.qKey}
              open={d.open}
              className="group border border-border-subtle rounded-lg bg-bg-secondary px-3 py-2"
            >
              <summary className="text-xs font-medium text-text-primary cursor-pointer select-none list-none flex items-center gap-2">
                <span className="text-text-tertiary transition-transform group-open:rotate-90" aria-hidden="true">
                  ›
                </span>
                {t(d.qKey)}
              </summary>
              <p className="text-[11px] text-text-secondary leading-relaxed mt-2 mb-1 pl-5">
                {t(d.aKey)}
              </p>
            </details>
          ))}
        </div>
      </Card>

      {/* Demo vs production assumptions */}
      <Card title={t('system.assumptions')} className="mb-3">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-text-tertiary border-b border-border-default">
                <th className="pb-2 pr-4 font-medium">{t('system.th.component')}</th>
                <th className="pb-2 pr-4 font-medium">{t('system.th.demo')}</th>
                <th className="pb-2 font-medium">{t('system.th.production')}</th>
              </tr>
            </thead>
            <tbody>
              {ASSUMPTION_ROWS.map((row) => (
                <tr key={row.c} className="border-b border-border-subtle last:border-0">
                  <td className="py-2 pr-4 text-text-primary font-medium whitespace-nowrap">{t(row.c)}</td>
                  <td className="py-2 pr-4 text-text-secondary">{t(row.d)}</td>
                  <td className="py-2 text-text-secondary">{t(row.p)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Live metrics */}
      <Card title={t('system.liveMetrics')}>
        <div className="grid grid-cols-4 gap-3">
          <div className="flex flex-col items-center py-2">
            <span className="text-[11px] text-text-tertiary uppercase tracking-wider">
              {t('system.metric.online')}
            </span>
            <span className="text-xl font-semibold font-mono text-text-primary mt-1">
              {summary ? `${summary.normal + summary.warning + summary.critical}/${summary.total}` : '—'}
            </span>
          </div>
          <div className="flex flex-col items-center py-2">
            <span className="text-[11px] text-text-tertiary uppercase tracking-wider">
              {t('system.metric.alarms')}
            </span>
            <span className="text-xl font-semibold font-mono text-text-primary mt-1">
              {summary ? summary.active_alarms : '—'}
            </span>
          </div>
          <div className="flex flex-col items-center py-2">
            <span className="text-[11px] text-text-tertiary uppercase tracking-wider">
              {t('system.metric.pending')}
            </span>
            <span className="text-xl font-semibold font-mono text-text-primary mt-1">
              {pendingApi.data ? pendingApi.data.length : '—'}
            </span>
          </div>
          <div className="flex flex-col items-center py-2">
            <span className="text-[11px] text-text-tertiary uppercase tracking-wider">
              {t('system.metric.savings')}
            </span>
            <span className="text-xl font-semibold font-mono text-success mt-1">
              {savingsApi.data
                ? `€${Math.max(savingsApi.data.total_savings_eur, 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                : '—'}
            </span>
          </div>
        </div>
      </Card>
    </div>
  )
}
