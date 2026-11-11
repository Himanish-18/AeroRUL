import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { DatasetId, PredictionRecord } from '../../lib/data/types';
import { loadBenchmarkData, LoadedBenchmarkData } from '../../lib/data/loader';
import {
  getMetricsForDataset,
  getPredictionsForDataset,
  getManifestForDataset,
} from '../../lib/data/adapter';
import { PrimaryMetricTile } from './PrimaryMetricTile';
import { SupportingMetricTiles } from './SupportingMetricTiles';
import { ModelMetadataCard } from './ModelMetadataCard';
import { ActualVsPredictedScatter } from './ActualVsPredictedScatter';
import { EnginePredictionTable } from './EnginePredictionTable';
import { EngineInspector } from './EngineInspector';
import { DashboardSkeleton } from '../../components/shared/Skeleton';
import { ErrorBanner } from '../../components/shared/ErrorBanner';

interface OverviewDashboardProps {
  selectedDataset: DatasetId;
}

export const OverviewDashboard: React.FC<OverviewDashboardProps> = ({
  selectedDataset,
}) => {
  const [data, setData] = useState<LoadedBenchmarkData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedUnitId, setSelectedUnitId] = useState<number | null>(null);

  const fetchData = useCallback(async (forceReload = false) => {
    try {
      setLoading(true);
      setError(null);
      const loaded = await loadBenchmarkData(forceReload);
      setData(loaded);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error loading research artifacts';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // When dataset changes, reset selected engine inspection
  useEffect(() => {
    setSelectedUnitId(null);
  }, [selectedDataset]);

  // Derive current subset records
  const currentMetrics = useMemo(() => {
    if (!data) return undefined;
    return getMetricsForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  const currentPredictions = useMemo(() => {
    if (!data) return [];
    return getPredictionsForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  const currentManifest = useMemo(() => {
    if (!data) return undefined;
    return getManifestForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  const inspectedPrediction = useMemo<PredictionRecord | null>(() => {
    if (selectedUnitId === null || currentPredictions.length === 0) return null;
    return currentPredictions.find((p) => p.unit_id === selectedUnitId) || null;
  }, [selectedUnitId, currentPredictions]);

  if (loading) {
    return <DashboardSkeleton />;
  }

  if (error || !data) {
    return (
      <ErrorBanner
        title="Failed to load research artifacts"
        message={error || 'Benchmark data could not be parsed.'}
        onRetry={() => fetchData(true)}
      />
    );
  }

  if (!currentMetrics || !currentManifest) {
    return (
      <ErrorBanner
        title={`Missing record for dataset ${selectedDataset}`}
        message="The requested dataset subset could not be located in the synchronized artifacts."
      />
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Overview Header Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2xs)' }}>
        <h1 style={{ fontSize: 'var(--text-display)', letterSpacing: 'var(--tracking-display)', margin: 0 }}>
          {selectedDataset} Benchmark Overview
        </h1>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-muted)', margin: 0, maxWidth: '65ch' }}>
          Official test endpoint performance of the frozen LightGBM model across{' '}
          {currentMetrics.n_official_test_engines} engine trajectories.
        </p>
      </div>

      {/* Metrics Row: Primary MAE tile + Supporting metrics grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 'var(--space-md)',
          alignItems: 'stretch',
        }}
      >
        <PrimaryMetricTile metrics={currentMetrics} />
        <SupportingMetricTiles metrics={currentMetrics} />
      </div>

      {/* Model Specification Card */}
      <ModelMetadataCard
        manifestConfig={currentManifest}
        frozenTimestamp={data.manifest.frozen_timestamp}
      />

      {/* Scatter Plot: Actual vs Predicted */}
      <ActualVsPredictedScatter
        predictions={currentPredictions}
        selectedUnitId={selectedUnitId}
        onSelectEngine={(unitId) => setSelectedUnitId(unitId)}
      />

      {/* Engine Detail Inspector (shown when an engine is selected) */}
      {inspectedPrediction && (
        <EngineInspector
          prediction={inspectedPrediction}
          onClose={() => setSelectedUnitId(null)}
        />
      )}

      {/* Engine Prediction Table */}
      <EnginePredictionTable
        predictions={currentPredictions}
        selectedUnitId={selectedUnitId}
        onSelectEngine={(unitId) => setSelectedUnitId(unitId)}
      />
    </div>
  );
};
