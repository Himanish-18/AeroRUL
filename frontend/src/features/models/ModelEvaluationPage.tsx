import React, { useState, useMemo } from 'react';
import { DatasetId, ModelEvaluationTab } from '../../lib/data/types';
import { useModelEvaluationData } from '../../lib/data/modelEvaluationLoader';
import { ModelEvaluationHeader } from './ModelEvaluationHeader';
import { MethodologyProvenanceNotice } from './MethodologyProvenanceNotice';
import { Phase5SelectionSection } from './Phase5SelectionSection';
import { Phase4BenchmarkSection } from './Phase4BenchmarkSection';
import { ConfigurationTuningSection } from './ConfigurationTuningSection';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ModelEvaluationPageProps {
  selectedDataset: DatasetId;
}

export const ModelEvaluationPage: React.FC<ModelEvaluationPageProps> = ({
  selectedDataset,
}) => {
  const [activeTab, setActiveTab] = useState<ModelEvaluationTab>('phase5_selection');
  const { data, loading, error, reload } = useModelEvaluationData();

  // Extract subset-specific slices
  const manifestConfig = useMemo(() => {
    return data?.manifest.selected_configurations[selectedDataset];
  }, [data, selectedDataset]);

  const p4Records = useMemo(() => {
    return data?.phase4Benchmark.filter((r) => r.dataset === selectedDataset) || [];
  }, [data, selectedDataset]);

  const p5CvRecords = useMemo(() => {
    return data?.phase5CrossValidation.filter((r) => r.dataset === selectedDataset) || [];
  }, [data, selectedDataset]);

  const p5DevValRecords = useMemo(() => {
    return data?.phase5DevelopmentValidation.filter((r) => r.dataset === selectedDataset) || [];
  }, [data, selectedDataset]);

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '360px',
          gap: 'var(--space-md)',
          color: 'var(--color-muted)',
        }}
      >
        <RefreshCw size={24} className="spin" color="var(--color-accent)" />
        <span style={{ fontSize: 'var(--text-sm)', fontFamily: 'var(--font-outlier)' }}>
          Loading candidate model evaluation artifacts…
        </span>
      </div>
    );
  }

  if (error || !data || !manifestConfig) {
    return (
      <div
        className="panel"
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '300px',
          gap: 'var(--space-md)',
          textAlign: 'center',
        }}
      >
        <AlertCircle size={32} color="var(--color-accent)" />
        <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>
          Failed to load model evaluation data
        </div>
        <p style={{ color: 'var(--color-muted)', fontSize: 'var(--text-xs)', maxWidth: '400px' }}>
          {error || `Missing configuration manifest for dataset ${selectedDataset}`}
        </p>
        <button type="button" className="btn btn-primary btn-sm" onClick={reload}>
          Retry loading
        </button>
      </div>
    );
  }

  const candidateCount = activeTab === 'phase4_benchmark' ? p4Records.length : p5CvRecords.length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Page Header */}
      <ModelEvaluationHeader
        selectedDataset={selectedDataset}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        candidateCount={candidateCount}
        frozenModel={manifestConfig.model_family}
        selectionCriterion={manifestConfig.selection_metric}
      />

      {/* Methodology & Provenance Notice */}
      <MethodologyProvenanceNotice />

      {/* Active Tab View */}
      {activeTab === 'phase5_selection' && (
        <Phase5SelectionSection
          dataset={selectedDataset}
          cvRecords={p5CvRecords}
          devValRecords={p5DevValRecords}
          manifestConfig={manifestConfig}
        />
      )}

      {activeTab === 'phase4_benchmark' && (
        <Phase4BenchmarkSection
          dataset={selectedDataset}
          records={p4Records}
        />
      )}

      {activeTab === 'configuration_tuning' && (
        <ConfigurationTuningSection
          dataset={selectedDataset}
          windowRecords={data.phase5WindowAblation}
          targetCapRecords={data.phase5TargetCapSensitivity}
          hyperparameterTrials={data.hyperparameterTrials}
          manifestConfig={manifestConfig}
        />
      )}
    </div>
  );
};
