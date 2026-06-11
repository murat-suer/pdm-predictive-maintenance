import { useCallback, useEffect, useRef, useState } from 'react'
import { apiGet, wsUrl } from './client'
import type { LiveMessage, LiveSnapshot } from './types'

interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

/** Fetch a GET endpoint, optionally re-fetching on an interval. */
export function useApi<T>(path: string, refreshMs?: number): ApiState<T> & { refetch: () => void } {
  const [state, setState] = useState<ApiState<T>>({ data: null, loading: true, error: null })

  const load = useCallback(() => {
    let cancelled = false
    apiGet<T>(path)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null })
      })
      .catch((err: Error) => {
        if (!cancelled) setState((prev) => ({ data: prev.data, loading: false, error: err.message }))
      })
    return () => {
      cancelled = true
    }
  }, [path])

  const [tick, setTick] = useState(0)
  const refetch = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    const cancel = load()
    if (!refreshMs) return cancel
    const interval = setInterval(load, refreshMs)
    return () => {
      cancel()
      clearInterval(interval)
    }
  }, [load, refreshMs, tick])

  return { ...state, refetch }
}

/**
 * Subscribe to /ws/live. Reconnects with backoff; onMessage fires for
 * every snapshot/anomaly. Returns the latest snapshot and connection state.
 */
export function useLiveSnapshot(onAnomaly?: (event: Record<string, string>) => void): {
  snapshot: LiveSnapshot | null
  connected: boolean
} {
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const onAnomalyRef = useRef(onAnomaly)
  useEffect(() => {
    onAnomalyRef.current = onAnomaly
  }, [onAnomaly])

  useEffect(() => {
    let socket: WebSocket | null = null
    let retryDelay = 1000
    let closed = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const connect = () => {
      if (closed) return
      socket = new WebSocket(wsUrl())
      socket.onopen = () => {
        setConnected(true)
        retryDelay = 1000
      }
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as LiveMessage
          if (message.type === 'snapshot') setSnapshot(message)
          else if (message.type === 'anomaly') onAnomalyRef.current?.(message.event)
        } catch {
          // malformed frame — ignore
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!closed) {
          retryTimer = setTimeout(connect, retryDelay)
          retryDelay = Math.min(retryDelay * 2, 15000)
        }
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      closed = true
      if (retryTimer) clearTimeout(retryTimer)
      socket?.close()
    }
  }, [])

  return { snapshot, connected }
}
