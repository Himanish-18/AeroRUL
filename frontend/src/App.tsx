import React, { useState, useEffect, useCallback } from 'react';
import { DatasetId, VALID_DATASETS, AppView, VALID_VIEWS } from './lib/data/types';
import { AppShell } from './components/layout/AppShell';
import { OverviewDashboard } from './features/overview/OverviewDashboard';
import { EnginePredictionsPage } from './features/predictions/EnginePredictionsPage';
import { ModelEvaluationPage } from './features/models/ModelEvaluationPage';

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

function getInitialView(): AppView {
  try {
    const params = new URLSearchParams(window.location.search);
    const v = params.get('view') as AppView | null;
    if (v && VALID_VIEWS.includes(v)) {
      return v;
    }
  } catch {
    // fallback to overview
  }
  return 'overview';
}

export const App: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>(getInitialDataset);
  const [currentView, setCurrentView] = useState<AppView>(getInitialView);

  // Sync state to URL query parameter
  const updateUrlParams = useCallback((view: AppView, dataset: DatasetId) => {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set('view', view);
      url.searchParams.set('dataset', dataset);
      window.history.pushState({}, '', url.toString());
    } catch {
      // Ignore if URL manipulations fail
    }
  }, []);

  const handleSelectDataset = useCallback((dataset: DatasetId) => {
    setSelectedDataset(dataset);
    updateUrlParams(currentView, dataset);
  }, [currentView, updateUrlParams]);

  const handleNavigate = useCallback((view: AppView) => {
    setCurrentView(view);
    updateUrlParams(view, selectedDataset);
  }, [selectedDataset, updateUrlParams]);

  // Sync state when browser forward/back button is used
  useEffect(() => {
    const onPopState = () => {
      setSelectedDataset(getInitialDataset());
      setCurrentView(getInitialView());
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  return (
    <AppShell
      currentView={currentView}
      onNavigate={handleNavigate}
      selectedDataset={selectedDataset}
      onSelectDataset={handleSelectDataset}
      frozenTimestamp="2026-10-27 00:35:50"
    >
      {currentView === 'models' ? (
        <ModelEvaluationPage selectedDataset={selectedDataset} />
      ) : currentView === 'predictions' ? (
        <EnginePredictionsPage selectedDataset={selectedDataset} />
      ) : (
        <OverviewDashboard selectedDataset={selectedDataset} />
      )}
    </AppShell>
  );
};
