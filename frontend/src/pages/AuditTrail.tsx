import { useState } from 'react'
import { Link } from 'react-router-dom'
import Card from '../components/Card'
import Badge from '../components/Badge'
import Button from '../components/Button'
import Header from '../components/Header'
import { useApi } from '../api/hooks'
import { useI18n } from '../i18n'
import type { AuditPage } from '../api/types'

const MACHINE_ID_PATTERN = /\b((?:AC|HX|CM)-\d{3})\b/

/** Render a machine ID inside free text as a link to its detail page. */
function TargetCell({ target }: { target: string }) {
  const match = target.match(MACHINE_ID_PATTERN)
  if (!match) return <>{target}</>
  const [before, after] = target.split(match[1], 2)
  return (
    <>
      {before}
      <Link to={`/machines/${match[1]}`} className="text-alarm-p0 hover:underline">
        {match[1]}
      </Link>
      {after}
    </>
  )
}

const categories = ['All', 'decision', 'alarm', 'system'] as const
const severities = ['All', 'info', 'warning', 'critical'] as const

const severityColor: Record<string, string> = {
  info: 'text-info',
  warning: 'text-alarm-p3',
  critical: 'text-alarm-p4',
}

const categoryIcon: Record<string, string> = {
  system: '⚙',
  alarm: '🔔',
  decision: '⚡',
}

export default function AuditTrail() {
  const { t } = useI18n()
  const [categoryFilter, setCategoryFilter] = useState<string>('All')
  const [severityFilter, setSeverityFilter] = useState<string>('All')
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(0)
  const perPage = 8

  const params = new URLSearchParams({ limit: '200' })
  if (categoryFilter !== 'All') params.set('category', categoryFilter)
  if (severityFilter !== 'All') params.set('severity', severityFilter)
  const auditApi = useApi<AuditPage>(`/audit/events?${params.toString()}`, 15000)

  const allEvents = auditApi.data?.events ?? []
  const filtered = allEvents.filter((e) => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      e.actor.toLowerCase().includes(q) ||
      e.action.toLowerCase().includes(q) ||
      e.target.toLowerCase().includes(q) ||
      e.id.toLowerCase().includes(q)
    )
  })

  const totalPages = Math.ceil(filtered.length / perPage)
  const paginated = filtered.slice(page * perPage, (page + 1) * perPage)

  return (
    <div className="p-4">
      <Header title={t('audit.title')} />

      {/* Filters */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <div className="flex gap-1 bg-bg-secondary rounded-md border border-border-default p-0.5">
          {categories.map((c) => (
            <button
              key={c}
              onClick={() => { setCategoryFilter(c); setPage(0) }}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                categoryFilter === c
                  ? 'bg-bg-elevated text-text-primary border border-border-default'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {c === 'All' ? t('audit.allCategories') : c.charAt(0).toUpperCase() + c.slice(1)}
            </button>
          ))}
        </div>

        <div className="flex gap-1 bg-bg-secondary rounded-md border border-border-default p-0.5">
          {severities.map((s) => (
            <button
              key={s}
              onClick={() => { setSeverityFilter(s); setPage(0) }}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                severityFilter === s
                  ? 'bg-bg-elevated text-text-primary border border-border-default'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {s === 'All' ? t('audit.allSeverities') : s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>

        <input
          type="text"
          value={searchQuery}
          onChange={(e) => { setSearchQuery(e.target.value); setPage(0) }}
          placeholder={t('audit.searchPlaceholder')}
          className="flex-1 min-w-[200px] bg-bg-secondary text-text-primary text-xs border border-border-default rounded px-3 py-1.5 placeholder:text-text-tertiary focus:outline-none focus:border-alarm-p0"
        />
      </div>

      {/* Audit Events */}
      <Card>
        {auditApi.error && allEvents.length === 0 ? (
          <p className="text-alarm-p4 text-xs py-6 text-center">{t('audit.backendUnreachable')} {auditApi.error}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-text-tertiary border-b border-border-default">
                  <th className="pb-2 pr-3 font-medium">ID</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.timestamp')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.category')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.severity')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.actor')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.action')}</th>
                  <th className="pb-2 pr-3 font-medium">{t('audit.col.target')}</th>
                  <th className="pb-2 font-medium">{t('audit.col.details')}</th>
                </tr>
              </thead>
              <tbody>
                {paginated.map((ev) => (
                  <tr key={ev.id} className="border-b border-border-subtle last:border-0">
                    <td className="py-2.5 pr-3 font-mono text-text-secondary whitespace-nowrap">{ev.id}</td>
                    <td className="py-2.5 pr-3 font-mono text-text-secondary whitespace-nowrap">
                      {new Date(ev.timestamp).toLocaleString()}
                    </td>
                    <td className="py-2.5 pr-3">
                      <span className="flex items-center gap-1 text-text-primary">
                        <span aria-hidden="true">{categoryIcon[ev.category] ?? '⚙'}</span>
                        <span className="capitalize">{ev.category}</span>
                      </span>
                    </td>
                    <td className="py-2.5 pr-3">
                      <span className={`flex items-center gap-1 ${severityColor[ev.severity]}`}>
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            ev.severity === 'critical'
                              ? 'bg-alarm-p4'
                              : ev.severity === 'warning'
                                ? 'bg-alarm-p3'
                                : 'bg-info'
                          }`}
                        />
                        <span className="capitalize">{ev.severity}</span>
                      </span>
                    </td>
                    <td className="py-2.5 pr-3 text-text-primary font-medium whitespace-nowrap">{ev.actor}</td>
                    <td className="py-2.5 pr-3">
                      <Badge
                        variant={
                          ev.severity === 'critical' ? 'error' : ev.severity === 'warning' ? 'warning' : 'info'
                        }
                        label={ev.action.replace(/_/g, ' ')}
                      />
                    </td>
                    <td className="py-2.5 pr-3 text-text-secondary whitespace-nowrap">
                      <TargetCell target={ev.target} />
                    </td>
                    <td className="py-2.5 text-text-secondary max-w-[240px] leading-relaxed">{ev.details ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!auditApi.loading && paginated.length === 0 && (
              <p className="text-text-tertiary text-center py-8 text-xs">
                {t('audit.noEvents')}
              </p>
            )}
            {auditApi.loading && allEvents.length === 0 && (
              <p className="text-text-tertiary text-center py-8 text-xs">{t('audit.loading')}</p>
            )}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-border-default">
            <span className="text-[11px] text-text-tertiary">
              {t('audit.showing', {
                from: page * perPage + 1,
                to: Math.min((page + 1) * perPage, filtered.length),
                total: filtered.length,
              })}
            </span>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}>
                {t('audit.prev')}
              </Button>
              <span className="text-xs text-text-secondary font-mono px-2">
                {page + 1}/{totalPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
              >
                {t('audit.next')}
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
