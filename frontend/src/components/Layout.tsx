import { useEffect, useState, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useApi, useLiveSnapshot } from '../api/hooks'
import type { FleetSummary } from '../api/types'
import { useI18n, type TranslationKey } from '../i18n'

interface LayoutProps {
  children: ReactNode
}

const navItems: { labelKey: TranslationKey; icon: string; href: string }[] = [
  { labelKey: 'nav.fleet', icon: '≡', href: '/fleet' },
  { labelKey: 'nav.decisions', icon: '⚡', href: '/decisions' },
  { labelKey: 'nav.reports', icon: '📊', href: '/reports' },
  { labelKey: 'nav.shiftHandover', icon: '⇄', href: '/shift-handover' },
  { labelKey: 'nav.auditTrail', icon: '📋', href: '/audit-trail' },
  { labelKey: 'nav.system', icon: '🛈', href: '/system' },
]

function loadTheme(): 'dark' | 'light' {
  return localStorage.getItem('pdm-theme') === 'light' ? 'light' : 'dark'
}

export default function Layout({ children }: LayoutProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>(loadTheme)
  const { lang, setLang, t } = useI18n()
  const summaryApi = useApi<FleetSummary>('/fleet/summary', 30000)
  const { snapshot, connected } = useLiveSnapshot()

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
    localStorage.setItem('pdm-theme', theme)
  }, [theme])

  // Prefer the live WebSocket snapshot over the REST poll for status dots.
  const counts = snapshot
    ? {
        normal: snapshot.machines.filter((m) => m.status === 'normal').length,
        warning: snapshot.machines.filter((m) => m.status === 'warning').length,
        critical: snapshot.machines.filter((m) => m.status === 'critical').length,
      }
    : summaryApi.data
      ? {
          normal: summaryApi.data.normal,
          warning: summaryApi.data.warning,
          critical: summaryApi.data.critical,
        }
      : null

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
        <header className="h-14 border-b border-border-default bg-bg-secondary flex items-center justify-between px-4 shrink-0">
          <div className="flex items-center gap-3">
            {/* Fleet status dots */}
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-success" title={t('chrome.normal')} />
              <span className="w-2 h-2 rounded-full bg-alarm-p3" title={t('chrome.warning')} />
              <span className="w-2 h-2 rounded-full bg-alarm-p4" title={t('chrome.critical')} />
            </div>
            <span className="text-[11px] text-text-tertiary border-l border-border-default pl-3">
              {counts
                ? `${counts.normal} ${t('chrome.normal')} · ${counts.warning} ${t('chrome.warning')} · ${counts.critical} ${t('chrome.critical')}`
                : t('chrome.connecting')}
            </span>
            {snapshot && snapshot.pending_decisions > 0 && (
              <NavLink
                to="/decisions"
                className="text-[11px] text-alarm-p3 border border-alarm-p3/40 bg-alarm-p3/10 px-2 py-0.5 rounded hover:bg-alarm-p3/20 transition-colors"
              >
                {snapshot.pending_decisions}{' '}
                {snapshot.pending_decisions > 1
                  ? t('chrome.pendingDecisions')
                  : t('chrome.pendingDecision')}
              </NavLink>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Live connection indicator */}
            <span className="flex items-center gap-1.5 text-[11px] text-text-tertiary">
              <span
                className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-success' : 'bg-text-tertiary'}`}
              />
              {connected ? t('chrome.live') : t('chrome.polling')}
            </span>

            {/* Role badge */}
            <span className="text-[11px] text-text-secondary bg-bg-hover px-2 py-0.5 rounded border border-border-subtle">
              {t('chrome.role')}
            </span>

            {/* Language toggle */}
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

            {/* Theme toggle */}
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
