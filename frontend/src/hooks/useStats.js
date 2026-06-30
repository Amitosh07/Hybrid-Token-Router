import { useCallback, useEffect, useState } from 'react';
import { fetchStats } from '../services/api.js';

const initialStats = {
  requests: 0,
  local: 0,
  fireworks: 0,
  money_saved: 0,
  latency: 0,
  tokens: 0
};

export function useStats({ onToast }) {
  const [stats, setStats] = useState(initialStats);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

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

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  return { stats, isLoading, error, refreshStats };
}
