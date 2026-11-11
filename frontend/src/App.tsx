import React, { useState, useEffect, useCallback } from 'react';
import { DatasetId, VALID_DATASETS } from './lib/data/types';
import { AppShell } from './components/layout/AppShell';
import { OverviewDashboard } from './features/overview/OverviewDashboard';

function getInitialDataset(): DatasetId {
  try {
    const params = new URLSearchParams(window.location.search);
    const ds = params.get('dataset') as DatasetId | null;
    if (ds && VALID_DATASETS.includes(ds)) {
      return ds;
    }
  } catch {
    // fallback to FD001 if window.location is unavailable
  }
  return 'FD001';
}

export const App: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>(getInitialDataset);

  // Sync state to URL query parameter
  const handleSelectDataset = useCallback((dataset: DatasetId) => {
    setSelectedDataset(dataset);
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('dataset', dataset);
      window.history.pushState({}, '', url.toString());
    } catch {
      // Ignore if URL manipulations fail
    }
  }, []);

  // Sync state when browser forward/back button is used
  useEffect(() => {
    const onPopState = () => {
      setSelectedDataset(getInitialDataset());
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  return (
    <AppShell
      selectedDataset={selectedDataset}
      onSelectDataset={handleSelectDataset}
      frozenTimestamp="2026-10-27 00:35:50"
    >
      <OverviewDashboard
        selectedDataset={selectedDataset}
      />
    </AppShell>
  );
};
