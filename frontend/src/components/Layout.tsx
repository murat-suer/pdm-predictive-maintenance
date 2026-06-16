import { useEffect, useState, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useApi, useLiveSnapshot } from '../api/hooks'
import type { FleetSummary, SavingsSummary } from '../api/types'
import { useI18n, type TranslationKey } from '../i18n'

interface LayoutProps {
  children: ReactNode
}

const navItems: { labelKey: TranslationKey; icon: string; href: string }[] = [
  { labelKey: 'nav.fleet', icon: '≡', href: '/fleet' },
  { labelKey: 'nav.decisions', icon: '⚡', href: '/decisions' },
  { labelKey: 'nav.analytics', icon: '📊', href: '/analytics' },
  { labelKey: 'nav.shiftHandover', icon: '⇄', href: '/shift-handover' },
  { labelKey: 'nav.auditTrail', icon: '📋', href: '/audit-trail' },
  { labelKey: 'nav.system', icon: '🛈', href: '/system' },
]

// ISA-101 tier shapes — must match FleetOverview / the spec table
const TIER_SHAPE: Record<string, string> = {
  normal:      '●',
  watch:       '▲',
  action:      '◆',
  critical:    '■',
  maintenance: '▣',
  offline:     '○',
}

function loadTheme(): 'dark' | 'light' {
  return localStorage.getItem('pdm-theme') === 'light' ? 'light' : 'dark'
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  if (days > 0) return `${days}d ${hours}h`
  const minutes = Math.floor((seconds % 3600) / 60)
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`
}

export default function Layout({ children }: LayoutProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>(loadTheme)
  const { lang, setLang, t } = useI18n()

  // Fleet summary — prefer live WS snapshot, fall back to REST poll
  const summaryApi = useApi<FleetSummary>('/fleet/summary', 15000)
  const savingsApi = useApi<SavingsSummary>('/savings', 30000)
  const { snapshot, connected } = useLiveSnapshot()

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
    localStorage.setItem('pdm-theme', theme)
  }, [theme])

  // Build live counts from WS snapshot; fall back to REST
  const counts = snapshot
    ? {
        normal:      snapshot.machines.filter((m) => m.status === 'normal').length,
        watch:       snapshot.machines.filter((m) => m.status === 'watch').length,
        action:      snapshot.machines.filter((m) => m.status === 'action').length,
        critical:    snapshot.machines.filter((m) => m.status === 'critical').length,
        maintenance: snapshot.machines.filter((m) => m.status === 'maintenance').length,
      }
    : summaryApi.data
      ? {
          normal:      summaryApi.data.normal,
          watch:       summaryApi.data.watch ?? 0,
          action:      summaryApi.data.action ?? 0,
          critical:    summaryApi.data.critical,
          maintenance: summaryApi.data.maintenance,
        }
      : null

  const savings = savingsApi.data

  return (
    <div className="min-h-screen bg-bg-primary flex">
      {/* Sidebar */}
      <aside className="w-56 border-r border-border-default bg-bg-secondary flex flex-col shrink-0">
        <div className="h-14 flex items-center px-4 border-b border-border-default">
          <span className="text-text-primary font-semibold text-sm tracking-tight">PDM-V3</span>
        </div>
        <nav className="flex-1 py-2">
          {navItems.map((item) => (
            <NavLink
              key={item.href}
              to={item.href}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors duration-150 ${
                  isActive
                    ? 'text-text-primary bg-bg-hover border-l-2 border-alarm-p0'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover border-l-2 border-transparent'
                }`
              }
            >
              <span className="text-base w-5 text-center">{item.icon}</span>
              <span>{t(item.labelKey)}</span>
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-border-default text-[11px] text-text-tertiary">
          v3.0.0
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Navbar */}
        <header className="h-14 border-b border-border-default bg-bg-secondary flex items-center justify-between px-4 shrink-0 gap-3">
          {/* Left: live fleet status summary — shape + colour + text (ISA-101, colour-blind safe) */}
          <div className="flex items-center gap-3 min-w-0 flex-1">
            {counts ? (
              <span className="text-[11px] text-text-tertiary flex items-center gap-2 flex-wrap">
                {/* Normal ● */}
                <span className="flex items-center gap-0.5">
                  <span className="text-success" aria-hidden="true">{TIER_SHAPE.normal}</span>
                  <span className="text-success font-medium">{counts.normal}</span>
                  <span className="text-text-tertiary ml-0.5">{t('chrome.normal')}</span>
                </span>
                <span className="text-border-default">·</span>
                {/* Watch ▲ */}
                <span className="flex items-center gap-0.5">
                  <span className="text-alarm-p3" aria-hidden="true">{TIER_SHAPE.watch}</span>
                  <span className="text-alarm-p3 font-medium">{counts.watch}</span>
                  <span className="text-text-tertiary ml-0.5">{t('chrome.watch')}</span>
                </span>
                <span className="text-border-default">·</span>
                {/* Action ◆ */}
                <span className="flex items-center gap-0.5">
                  <span style={{ color: '#fb923c' }} aria-hidden="true">{TIER_SHAPE.action}</span>
                  <span className="font-medium" style={{ color: '#fb923c' }}>{counts.action}</span>
                  <span className="text-text-tertiary ml-0.5">{t('chrome.action')}</span>
                </span>
                <span className="text-border-default">·</span>
                {/* Critical ■ */}
                <span className="flex items-center gap-0.5">
                  <span className="text-alarm-p4" aria-hidden="true">{TIER_SHAPE.critical}</span>
                  <span className="text-alarm-p4 font-medium">{counts.critical}</span>
                  <span className="text-text-tertiary ml-0.5">{t('chrome.critical')}</span>
                </span>
                {/* Maintenance ▣ — only if non-zero */}
                {counts.maintenance > 0 && (
                  <>
                    <span className="text-border-default">·</span>
                    <span className="flex items-center gap-0.5">
                      <span className="text-alarm-p0" aria-hidden="true">{TIER_SHAPE.maintenance}</span>
                      <span className="text-alarm-p0 font-medium">{counts.maintenance}</span>
                      <span className="text-text-tertiary ml-0.5">{t('chrome.maintenance')}</span>
                    </span>
                  </>
                )}
              </span>
            ) : (
              <span className="text-[11px] text-text-tertiary">{t('chrome.connecting')}</span>
            )}

            {/* Pending decisions badge */}
            {snapshot && snapshot.pending_decisions > 0 && (
              <NavLink
                to="/decisions"
                className="text-[11px] text-alarm-p3 border border-alarm-p3/40 bg-alarm-p3/10 px-2 py-0.5 rounded hover:bg-alarm-p3/20 transition-colors shrink-0"
              >
                {snapshot.pending_decisions}{' '}
                {snapshot.pending_decisions > 1
                  ? t('chrome.pendingDecisions')
                  : t('chrome.pendingDecision')}
              </NavLink>
            )}
          </div>

          {/* Centre: Savings KPI — persistent on every tab */}
          <div className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded border border-border-subtle bg-bg-elevated shrink-0">
            <span className="text-[10px] text-text-tertiary uppercase tracking-wider">{t('fleet.savings')}</span>
            <span className="text-success font-semibold font-mono text-[12px]">
              {savings
                ? `€${Math.max(savings.total_savings_eur, 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                : '—'}
            </span>
            {savings && (
              <span className="text-[10px] text-text-tertiary border-l border-border-subtle pl-1.5">
                {t('fleet.savingsWindow')}
                {savings.uptime_seconds != null
                  ? ` · ${t('fleet.uptimeReset', { up: formatUptime(savings.uptime_seconds) })}`
                  : ''}
              </span>
            )}
          </div>

          {/* Right: connection, role, lang, theme */}
          <div className="flex items-center gap-3 shrink-0">
            <span className="flex items-center gap-1.5 text-[11px] text-text-tertiary">
              <span
                className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-success' : 'bg-text-tertiary'}`}
              />
              {connected ? t('chrome.live') : t('chrome.polling')}
            </span>

            <span className="text-[11px] text-text-secondary bg-bg-hover px-2 py-0.5 rounded border border-border-subtle">
              {t('chrome.role')}
            </span>

            <div className="flex rounded border border-border-subtle overflow-hidden text-[11px]">
              {(['en', 'de', 'tr'] as const).map((code) => (
                <button
                  key={code}
                  onClick={() => setLang(code)}
                  aria-pressed={lang === code}
                  className={`px-2 py-0.5 uppercase transition-colors ${
                    lang === code
                      ? 'bg-bg-elevated text-text-primary font-semibold'
                      : 'text-text-tertiary hover:text-text-primary'
                  }`}
                >
                  {code}
                </button>
              ))}
            </div>

            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="text-text-secondary hover:text-text-primary transition-colors text-sm px-2 py-1 rounded hover:bg-bg-hover"
              title={t('chrome.toggleTheme')}
            >
              {theme === 'dark' ? '☀' : '☾'}
            </button>
          </div>
        </header>

        {/* Main content */}
        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  )
}
