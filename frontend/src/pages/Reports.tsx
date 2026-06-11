import { useState } from 'react'
import Card from '../components/Card'
import Header from '../components/Header'
import AuditTrail from './AuditTrail'
import ShiftHandover from './ShiftHandover'
import { useI18n } from '../i18n'

type ReportTab = 'audit' | 'handover'

export default function Reports() {
  const { t } = useI18n()
  const [activeTab, setActiveTab] = useState<ReportTab>('audit')

  const tabs: { id: ReportTab; label: string; icon: string }[] = [
    { id: 'audit', label: t('nav.auditTrail'), icon: '📋' },
    { id: 'handover', label: t('nav.shiftHandover'), icon: '⇄' },
  ]

  return (
    <div className="p-4">
      <Header title={t('reports.title')} />

      {/* Tab Navigation */}
      <Card className="mb-3">
        <div className="flex gap-1">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-medium transition-all duration-150 min-h-10 ${
                  isActive
                    ? 'bg-bg-hover text-text-primary border border-border-default'
                    : 'text-text-tertiary hover:text-text-secondary hover:bg-bg-hover/50'
                }`}
              >
                <span aria-hidden="true">{tab.icon}</span>
                {tab.label}
              </button>
            )
          })}
        </div>
      </Card>

      {/* Tab Content */}
      {activeTab === 'audit' && <AuditTrail />}
      {activeTab === 'handover' && <ShiftHandover />}
    </div>
  )
}
