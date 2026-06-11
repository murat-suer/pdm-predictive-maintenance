import { useState } from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Header from '../components/Header'
import { useApi } from '../api/hooks'
import { useI18n } from '../i18n'
import type { AlarmItem, ShiftReportItem } from '../api/types'

const checklistKeys = ['shift.check1', 'shift.check2', 'shift.check3', 'shift.check4', 'shift.check5'] as const

function levelToPriority(level: number, severity: string): 'p1' | 'p2' | 'p3' | 'p4' {
  if (severity === 'CRITICAL') return 'p4'
  if (level >= 2) return 'p3'
  return 'p2'
}

function formatDuration(minutes: number): string {
  const h = Math.floor(minutes / 60)
  const m = minutes % 60
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

export default function ShiftHandover() {
  const { t } = useI18n()
  const alarmsApi = useApi<AlarmItem[]>('/alarms?active=true', 10000)
  const reportsApi = useApi<ShiftReportItem[]>('/shift-reports?limit=4', 60000)
  const [selectedReport, setSelectedReport] = useState<number | null>(null)
  const [acknowledgedItems, setAcknowledgedItems] = useState<Set<number>>(new Set())

  const activeAlarms = alarmsApi.data ?? []
  const shiftReports = reportsApi.data ?? []

  const toggleChecklist = (idx: number) => {
    setAcknowledgedItems((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  return (
    <div className="p-4">
      <Header title={t('shift.title')} />

      {/* Active Alarm Handover Banner */}
      <Card className={`mb-3 ${activeAlarms.length > 0 ? 'border-alarm-p4/50' : ''}`}>
        <div className="flex items-center gap-2 mb-3">
          <span
            className={`w-2 h-2 rounded-full ${activeAlarms.length > 0 ? 'bg-alarm-p4 animate-pulse' : 'bg-success'}`}
          />
          <span className="text-sm font-medium text-text-primary">
            {activeAlarms.length > 0 ? t('shift.activeAlarmsHandover') : t('shift.noActiveAlarms')}
          </span>
          <Badge
            variant={activeAlarms.length > 0 ? 'error' : 'success'}
            label={`${activeAlarms.length} ${t('shift.active')}`}
          />
        </div>
        {alarmsApi.error && activeAlarms.length === 0 ? (
          <p className="text-alarm-p4 text-xs">{t('shift.backendUnreachable')} {alarmsApi.error}</p>
        ) : activeAlarms.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-text-tertiary border-b border-border-default">
                  <th className="pb-2 pr-3 font-medium">ID</th>
                  <th className="pb-2 pr-3 font-medium">{t('shift.col.machine')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('shift.col.alarm')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('shift.col.state')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('shift.col.priority')}</th>
                  <th className="pb-2 font-medium">{t('shift.col.duration')}</th>
                </tr>
              </thead>
              <tbody>
                {activeAlarms.map((a) => (
                  <tr key={a.id} className="border-b border-border-subtle last:border-0">
                    <td className="py-2 pr-3 font-mono text-text-secondary">AL-{a.id}</td>
                    <td className="py-2 pr-3 font-medium">
                      <Link to={`/machines/${a.machine_id}`} className="text-alarm-p0 hover:underline">
                        {a.machine_id}
                      </Link>
                    </td>
                    <td className="py-2 pr-3 text-text-primary">
                      {(a.fault_type ?? a.top_contributing_sensor ?? 'Anomaly').replace(/_/g, ' ')}
                    </td>
                    <td className="py-2 pr-3 text-text-secondary capitalize">
                      {a.status.replace(/_/g, ' ').toLowerCase()}
                    </td>
                    <td className="py-2 pr-3">
                      <Badge variant={levelToPriority(a.level, a.severity)} />
                    </td>
                    <td className="py-2 font-mono text-text-secondary">{formatDuration(a.duration_minutes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-text-tertiary text-xs">{t('shift.allNominal')}</p>
        )}
      </Card>

      <div className="grid grid-cols-5 gap-3">
        {/* Shift Reports — spans 3 cols */}
        <div className="col-span-3">
          <Card title={t('shift.reports')} subtitle={t('shift.reportsSub')}>
            {shiftReports.length === 0 ? (
              <p className="text-text-tertiary text-xs py-4">
                {t('shift.noReports')}
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {shiftReports.map((report) => {
                  const isSelected = selectedReport === report.id
                  const data = report.report_data as {
                    summary?: string
                    critical_count?: number
                    warning_count?: number
                  }
                  return (
                    <button
                      key={report.id}
                      onClick={() => setSelectedReport(isSelected ? null : report.id)}
                      className={`w-full text-left p-3 rounded-lg border transition-all duration-150 ${
                        isSelected
                          ? 'border-alarm-p0 bg-alarm-p0/10'
                          : 'border-border-default bg-bg-secondary hover:bg-bg-hover'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="text-text-primary text-xs font-medium">
                            SH-{report.id} · {t('shift.shift')} {report.shift_type}
                          </span>
                          <span className="text-[11px] text-text-tertiary font-mono">
                            {new Date(report.shift_start).toLocaleString()}
                          </span>
                        </div>
                        <Badge variant="info" label={report.shift_type} />
                      </div>
                      {data.summary && (
                        <p className="text-[11px] text-text-secondary leading-relaxed mb-1.5">{data.summary}</p>
                      )}
                      <div className="flex items-center gap-2">
                        {(data.critical_count ?? 0) > 0 && (
                          <span className="text-[10px] text-alarm-p4 font-mono">{data.critical_count} {t('shift.critical')}</span>
                        )}
                        {(data.warning_count ?? 0) > 0 && (
                          <span className="text-[10px] text-alarm-p3 font-mono">{data.warning_count} {t('shift.warning')}</span>
                        )}
                      </div>
                      {isSelected && (
                        <pre className="mt-2 text-[10px] text-text-tertiary bg-bg-primary rounded p-2 overflow-x-auto">
                          {JSON.stringify(report.report_data, null, 2)}
                        </pre>
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </Card>
        </div>

        {/* Handover Checklist — spans 2 cols */}
        <div className="col-span-2">
          <Card title={t('shift.checklist')} subtitle={t('shift.checklistSub')}>
            <div className="flex flex-col gap-2">
              {checklistKeys.map((item, idx) => {
                const checked = acknowledgedItems.has(idx)
                return (
                  <button
                    key={idx}
                    onClick={() => toggleChecklist(idx)}
                    className={`flex items-center gap-3 p-2.5 rounded-lg border transition-all duration-150 text-left ${
                      checked
                        ? 'border-success/40 bg-success/5'
                        : 'border-border-default bg-bg-secondary hover:bg-bg-hover'
                    }`}
                  >
                    <span
                      className={`w-4 h-4 rounded border flex items-center justify-center text-[10px] shrink-0 transition-colors ${
                        checked ? 'bg-success border-success text-white' : 'border-border-default bg-transparent'
                      }`}
                    >
                      {checked ? '✓' : ''}
                    </span>
                    <span className={`text-xs ${checked ? 'text-text-secondary line-through' : 'text-text-primary'}`}>
                      {t(item)}
                    </span>
                  </button>
                )
              })}
            </div>

            <div className="mt-4 pt-3 border-t border-border-default">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-text-tertiary">
                  {t('shift.acknowledged', { done: acknowledgedItems.size, total: checklistKeys.length })}
                </span>
                <span className="text-[11px] font-mono text-text-primary">
                  {Math.round((acknowledgedItems.size / checklistKeys.length) * 100)}%
                </span>
              </div>
              <div className="h-1.5 bg-bg-secondary rounded-full overflow-hidden border border-border-subtle">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${(acknowledgedItems.size / checklistKeys.length) * 100}%`,
                    backgroundColor: acknowledgedItems.size === checklistKeys.length ? '#00e676' : '#448aff',
                  }}
                />
              </div>
            </div>

            <Button
              variant="primary"
              size="md"
              className="w-full mt-3"
              disabled={acknowledgedItems.size !== checklistKeys.length}
            >
              {acknowledgedItems.size === checklistKeys.length
                ? t('shift.completeHandover')
                : t('shift.acknowledgeAll')}
            </Button>
          </Card>
        </div>
      </div>
    </div>
  )
}
