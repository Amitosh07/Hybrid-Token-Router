import { useCallback, useEffect, useState } from 'react';
import { fetchStats } from '../services/api.js';

const initialStats = {
  current_provider: 'none',
  total_requests: 0,
  local_requests: 0,
  remote_requests: 0,
  fallback_count: 0,
  average_latency_ms: 0,
  average_confidence: 0,
  router_version: '--',
  uptime_seconds: 0,
  timestamp: null,
};

export function useStats({ onToast }) {
  const [stats, setStats] = useState(initialStats);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Full refresh: shows loading state + error toast. Used on mount and manual Refresh button.
  const refreshStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchStats();
      setStats((current) => ({ ...current, ...data }));
    } catch (err) {
      setError(err);
      onToast?.('Unable to load dashboard stats.', 'error');
    } finally {
      setIsLoading(false);
    }
  }, [onToast]);

  // Silent refresh: no loading spinner, no error toast.
  // Called automatically after every successful chat prompt so the dashboard
  // stays in sync without any visible flicker.
  const silentRefresh = useCallback(async () => {
    try {
      const data = await fetchStats();
      setStats((current) => ({ ...current, ...data }));
    } catch {
      // Swallow silently — a failed background sync is not user-visible
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  return { stats, isLoading, error, refreshStats, silentRefresh };
}
