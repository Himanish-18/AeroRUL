import React, { useState, useMemo } from 'react';
import { DatasetId, PredictionFilterState } from '../../lib/data/types';
import { useBenchmarkData } from '../../lib/data/loader';
import {
  getMetricsForDataset,
  getPredictionsForDataset,
  getManifestForDataset,
  getPredictionSummary,
  filterAndSortPredictionsAdvanced,
} from '../../lib/data/adapter';
import { PredictionSummaryCards } from './PredictionSummaryCards';
import { PredictionFilterToolbar } from './PredictionFilterToolbar';
import { ErrorDistributionChart } from './ErrorDistributionChart';
import { TopOutliersCard } from './TopOutliersCard';
import { EngineEndpointInspector } from './EngineEndpointInspector';
import { PredictionsDetailedTable } from './PredictionsDetailedTable';
import { ActualVsPredictedScatter } from '../overview/ActualVsPredictedScatter';
import { Skeleton } from '../../components/shared/Skeleton';
import { ErrorBanner } from '../../components/shared/ErrorBanner';
import { formatCycles } from '../../lib/utils/format';
import { Layers, ShieldCheck, Cpu } from 'lucide-react';

interface EnginePredictionsPageProps {
  selectedDataset: DatasetId;
}

export const EnginePredictionsPage: React.FC<EnginePredictionsPageProps> = ({ selectedDataset }) => {
  const { data, loading, error, reload } = useBenchmarkData();
  const [selectedUnitId, setSelectedUnitId] = useState<number | null>(null);

  const [filters, setFilters] = useState<PredictionFilterState>({
    searchQuery: '',
    errorClassification: 'all',
    minAbsoluteError: 0,
  });

  const datasetMetrics = useMemo(() => {
    if (!data) return undefined;
    return getMetricsForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  const datasetManifest = useMemo(() => {
    if (!data) return undefined;
    return getManifestForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  const allPredictions = useMemo(() => {
    if (!data) return [];
    return getPredictionsForDataset(data, selectedDataset);
  }, [data, selectedDataset]);

  // Task 5: Reset selected unit unconditionally whenever selectedDataset changes
  React.useEffect(() => {
    setSelectedUnitId(null);
  }, [selectedDataset]);

  // Overall dataset summary stats (unfiltered baseline)
  const summary = useMemo(() => {
    return getPredictionSummary(allPredictions, datasetMetrics);
  }, [allPredictions, datasetMetrics]);

  // Filtered predictions
  const filteredPredictions = useMemo(() => {
    return filterAndSortPredictionsAdvanced(
      allPredictions,
      filters,
      { field: 'unit_id', direction: 'asc' }
    );
  }, [allPredictions, filters]);

  // Selected engine record for inspector
  const selectedPrediction = useMemo(() => {
    if (selectedUnitId === null) return null;
    return allPredictions.find((p) => p.unit_id === selectedUnitId) || null;
  }, [allPredictions, selectedUnitId]);

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <Skeleton width="40%" height="24px" />
          <Skeleton width="70%" height="16px" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--space-md)' }}>
          <Skeleton height="110px" />
          <Skeleton height="110px" />
          <Skeleton height="110px" />
          <Skeleton height="110px" />
        </div>
        <Skeleton height="380px" />
        <Skeleton height="420px" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <ErrorBanner
        message={error || 'Failed to load official benchmark predictions.'}
        onRetry={reload}
      />
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Workspace Header & Frozen Configuration Strip */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <h2 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
                Engine Predictions Workspace: {selectedDataset}
              </h2>
              <span className="chip chip-cool">Official Test</span>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Individual endpoint Remaining Useful Life (RUL) predictions, error distributions, and post-processing decomposition.
            </p>
          </div>

          {datasetManifest && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', flexWrap: 'wrap' }}>
              <span className="chip" style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink)' }}>
                <Cpu size={12} color="var(--color-accent)" />
                <span>{datasetManifest.model_family}</span>
              </span>
              <span className="chip" style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink)' }}>
                <Layers size={12} color="var(--color-cool)" />
                <span>Cap: {datasetManifest.target_cap}</span>
              </span>
              <span className="chip" style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                <ShieldCheck size={12} color="var(--color-ok)" />
                <span>CV MAE: {formatCycles(datasetManifest.inner_cv_scores.pooled_mae)}</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Summary Stat Cards */}
      <PredictionSummaryCards summary={summary} metric={datasetMetrics} />

      {/* Outlier Outliers Panel */}
      <TopOutliersCard
        predictions={allPredictions}
        selectedUnitId={selectedUnitId}
        onSelectEngine={(id) => setSelectedUnitId(id)}
      />

      {/* Engine Inspector Drawer / Detail Panel */}
      {selectedPrediction && (
        <EngineEndpointInspector
          prediction={selectedPrediction}
          onClose={() => setSelectedUnitId(null)}
        />
      )}

      {/* Dual Visualizations Grid: Scatter Plot & Signed Error Histogram */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: 'var(--space-lg)',
        }}
      >
        <ActualVsPredictedScatter
          predictions={filteredPredictions}
          selectedUnitId={selectedUnitId}
          onSelectEngine={(id) => setSelectedUnitId(id)}
        />

        <ErrorDistributionChart
          predictions={filteredPredictions}
          isFiltered={filteredPredictions.length !== allPredictions.length}
          totalDatasetCount={allPredictions.length}
        />
      </div>

      {/* Filter Toolbar */}
      <PredictionFilterToolbar
        filters={filters}
        onFilterChange={setFilters}
        totalEngines={allPredictions.length}
        filteredCount={filteredPredictions.length}
        overCount={summary.overpredictionCount}
        underCount={summary.underpredictionCount}
      />

      {/* Predictions Detailed Table */}
      <PredictionsDetailedTable
        predictions={filteredPredictions}
        selectedUnitId={selectedUnitId}
        onSelectEngine={(id) => setSelectedUnitId(id)}
        onResetFilters={() =>
          setFilters({
            searchQuery: '',
            errorClassification: 'all',
            minAbsoluteError: 0,
          })
        }
      />
    </div>
  );
};
