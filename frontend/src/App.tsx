import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import FleetOverview from './pages/FleetOverview'
import DecisionScreen from './pages/DecisionScreen'
import MachineDetail from './pages/MachineDetail'
import Analytics from './pages/Analytics'
import ShiftHandover from './pages/ShiftHandover'
import AuditTrail from './pages/AuditTrail'
import SystemOverview from './pages/SystemOverview'
import { I18nProvider } from './i18n'

export default function App() {
  return (
    <BrowserRouter>
      <I18nProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/fleet" replace />} />
            <Route path="/fleet" element={<FleetOverview />} />
            <Route path="/machines/:id" element={<MachineDetail />} />
            <Route path="/decisions" element={<DecisionScreen />} />
            {/* /reports redirects to /analytics for any bookmarked links */}
            <Route path="/reports" element={<Navigate to="/analytics" replace />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/shift-handover" element={<ShiftHandover />} />
            <Route path="/audit-trail" element={<AuditTrail />} />
            <Route path="/system" element={<SystemOverview />} />
          </Routes>
        </Layout>
      </I18nProvider>
    </BrowserRouter>
  )
}
