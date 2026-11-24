import React, { useState } from 'react';
import { DatasetId } from '../../lib/data/types';
import { useFeatureIntelligenceData } from '../../lib/data/featureIntelligenceLoader';
import { FeatureImportanceSection } from './FeatureImportanceSection';
import { SensorDegradationSection } from './SensorDegradationSection';
import { Cpu, Activity, BarChart2, Layers, AlertCircle, RefreshCw } from 'lucide-react';

interface FeatureIntelligencePageProps {
  selectedDataset: DatasetId;
}

export const FeatureIntelligencePage: React.FC<FeatureIntelligencePageProps> = ({
  selectedDataset,
}) => {
  const [activeTab, setActiveTab] = useState<'importance' | 'degradation'>('importance');
  const { data, trajectories, isLoading, isLoadingTrajectories, error, reload } =
    useFeatureIntelligenceData(selectedDataset);

  if (isLoading && !data) {
    return (
      <div
        className="app-content-inner"
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 400,
          gap: 'var(--space-md)',
          color: 'var(--color-muted)',
        }}
      >
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <p style={{ fontSize: 'var(--text-sm)', margin: 0 }}>
          Loading Feature Intelligence and Degradation artifacts for {selectedDataset}…
        </p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div
        className="app-content-inner"
        style={{
          padding: 'var(--space-xl)',
          maxWidth: 600,
          margin: '0 auto',
          textAlign: 'center',
        }}
      >
        <div className="card" style={{ padding: 'var(--space-xl)', borderLeft: '4px solid var(--color-error)' }}>
          <AlertCircle size={32} style={{ color: 'var(--color-error)', margin: '0 auto var(--space-md)' }} />
          <h3 style={{ fontSize: 'var(--text-base)', color: 'var(--color-ink)', margin: '0 0 var(--space-xs)' }}>
            Failed to Load Feature Intelligence Data
          </h3>
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-muted)', marginBottom: 'var(--space-md)' }}>
            {error}
          </p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={reload}
            style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-xs)' }}
          >
            <RefreshCw size={14} />
            Retry Synchronization
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const features = data.featureImportance[selectedDataset] || [];
  const sensors = data.sensorImportance[selectedDataset] || [];
  const manifestEntry = data.featureManifest.datasets[selectedDataset];

  const topFeature = features[0];
  const topSensor = sensors[0];

  return (
    <div
      className="app-content-inner"
      style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}
      data-testid="feature-intelligence-page"
    >
      {/* Overview Metric Banner */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        <div
          className="card"
          style={{
            padding: 'var(--space-md)',
            background: 'var(--color-paper-2)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Cpu size={14} style={{ color: 'var(--color-accent)' }} />
            Model & Feature Space
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-ink)',
              marginTop: 4,
            }}
          >
            {manifestEntry?.n_features ?? features.length}{' '}
            <span style={{ fontSize: 'var(--text-xs)', fontWeight: 400, color: 'var(--color-muted)' }}>features</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            Frozen {manifestEntry?.model_family ?? 'LightGBM'} · {manifestEntry?.feature_configuration ?? 'ALL_WINDOWS'}
          </div>
        </div>

        <div
          className="card"
          style={{
            padding: 'var(--space-md)',
            background: 'var(--color-paper-2)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <BarChart2 size={14} style={{ color: 'var(--color-accent)' }} />
            Dominant Feature
          </div>
          <div
            style={{
              fontSize: 'var(--text-base)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-accent)',
              marginTop: 4,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={topFeature?.feature_name}
          >
            {topFeature ? topFeature.feature_name : '—'}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            {topFeature ? `${topFeature.gain_importance_pct.toFixed(2)}% total gain (#1)` : '—'}
          </div>
        </div>

        <div
          className="card"
          style={{
            padding: 'var(--space-md)',
            background: 'var(--color-paper-2)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Activity size={14} style={{ color: 'var(--color-cool)' }} />
            Dominant Sensor
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-ink)',
              marginTop: 4,
            }}
          >
            {topSensor ? topSensor.sensor : '—'}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            {topSensor ? `${topSensor.gain_importance_pct.toFixed(2)}% model gain (${topSensor.feature_count} feats)` : '—'}
          </div>
        </div>

        <div
          className="card"
          style={{
            padding: 'var(--space-md)',
            background: 'var(--color-paper-2)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Layers size={14} style={{ color: 'var(--color-ok)' }} />
            Tree Decision Splits
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-ink)',
              marginTop: 4,
            }}
          >
            {manifestEntry?.total_split_importance.toLocaleString() ?? '—'}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            Ensemble decision nodes
          </div>
        </div>
      </div>

      {/* Sub-Workspace Tab Navigation */}
      <div style={{ display: 'flex', borderBottom: 'var(--rule-hair) solid var(--color-rule)', gap: 'var(--space-md)' }}>
        <button
          type="button"
          onClick={() => setActiveTab('importance')}
          style={{
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'importance' ? '2px solid var(--color-accent)' : '2px solid transparent',
            padding: 'var(--space-sm) var(--space-md)',
            color: activeTab === 'importance' ? 'var(--color-accent)' : 'var(--color-muted)',
            fontWeight: activeTab === 'importance' ? 600 : 400,
            fontSize: 'var(--text-sm)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            whiteSpace: 'nowrap',
          }}
          aria-selected={activeTab === 'importance'}
          role="tab"
          id="tab-feature-importance"
        >
          <BarChart2 size={16} />
          Feature Importance
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('degradation')}
          style={{
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'degradation' ? '2px solid var(--color-accent)' : '2px solid transparent',
            padding: 'var(--space-sm) var(--space-md)',
            color: activeTab === 'degradation' ? 'var(--color-accent)' : 'var(--color-muted)',
            fontWeight: activeTab === 'degradation' ? 600 : 400,
            fontSize: 'var(--text-sm)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            whiteSpace: 'nowrap',
          }}
          aria-selected={activeTab === 'degradation'}
          role="tab"
          id="tab-sensor-degradation"
        >
          <Activity size={16} />
          Sensor Degradation
        </button>
      </div>

      {/* Tab Panels */}
      {activeTab === 'importance' ? (
        <FeatureImportanceSection
          dataset={selectedDataset}
          features={features}
          familyRecords={data.familyBreakdown}
          trajectoriesManifest={data.trajectoriesManifest}
        />
      ) : (
        <SensorDegradationSection
          dataset={selectedDataset}
          trajectories={trajectories}
          isLoadingTrajectories={isLoadingTrajectories}
          manifest={data.trajectoriesManifest}
        />
      )}
    </div>
  );
};
