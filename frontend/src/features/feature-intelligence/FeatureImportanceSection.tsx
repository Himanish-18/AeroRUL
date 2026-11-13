import React, { useState, useMemo } from 'react';
import {
  DatasetId,
  FeatureImportanceRecord,
  FeatureFamilyRecord,
  SensorTrajectoriesManifest,
} from '../../lib/data/types';
import { getFilteredFeatures } from '../../lib/data/featureIntelligenceAdapter';
import { FeatureImportanceChart } from './FeatureImportanceChart';
import { FeatureDetailInspector } from './FeatureDetailInspector';
import { FeatureFamilyBreakdown } from './FeatureFamilyBreakdown';
import { ImportanceVsCorrelation } from './ImportanceVsCorrelation';
import { Search, X, BarChart3, Hash } from 'lucide-react';

interface FeatureImportanceSectionProps {
  dataset: DatasetId;
  features: FeatureImportanceRecord[];
  familyRecords: FeatureFamilyRecord[];
  trajectoriesManifest?: SensorTrajectoriesManifest | null;
}

export const FeatureImportanceSection: React.FC<FeatureImportanceSectionProps> = ({
  dataset,
  features,
  familyRecords,
  trajectoriesManifest,
}) => {
  const [metric, setMetric] = useState<'gain' | 'split'>('gain');
  const [topN, setTopN] = useState<number | 'all'>(15);
  const [search, setSearch] = useState<string>('');
  const [selectedFeatureName, setSelectedFeatureName] = useState<string | null>(null);

  // Filtered list
  const filteredFeatures = useMemo(() => {
    return getFilteredFeatures(features, metric, topN, search);
  }, [features, metric, topN, search]);

  // Selected feature object (defaults to top feature in current filtered list or overall top)
  const selectedFeature = useMemo(() => {
    if (selectedFeatureName) {
      const found = features.find((f) => f.feature_name === selectedFeatureName);
      if (found) return found;
    }
    return filteredFeatures[0] || features[0] || null;
  }, [features, filteredFeatures, selectedFeatureName]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Controls Bar */}
      <div
        className="card"
        style={{
          padding: 'var(--space-md) var(--space-lg)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 'var(--space-md)',
          background: 'var(--color-paper-2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          {/* Importance Metric Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Metric:</span>
            <div style={{ display: 'flex', background: 'var(--color-paper-3)', padding: 2, borderRadius: 'var(--radius-sm)' }}>
              <button
                type="button"
                className={`btn btn-sm ${metric === 'gain' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setMetric('gain')}
                style={{ fontSize: 'var(--text-xs)', padding: '3px 10px', display: 'flex', alignItems: 'center', gap: 4 }}
              >
                <BarChart3 size={13} />
                Gain Importance
              </button>
              <button
                type="button"
                className={`btn btn-sm ${metric === 'split' ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setMetric('split')}
                style={{ fontSize: 'var(--text-xs)', padding: '3px 10px', display: 'flex', alignItems: 'center', gap: 4 }}
              >
                <Hash size={13} />
                Split Frequency
              </button>
            </div>
          </div>

          {/* Top-N Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Show:</span>
            <div style={{ display: 'flex', background: 'var(--color-paper-3)', padding: 2, borderRadius: 'var(--radius-sm)' }}>
              {([10, 15, 25, 'all'] as const).map((n) => (
                <button
                  key={n}
                  type="button"
                  className={`btn btn-sm ${topN === n ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => setTopN(n)}
                  style={{ fontSize: 'var(--text-xs)', padding: '3px 8px' }}
                >
                  {n === 'all' ? 'All' : `Top ${n}`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Search Input */}
        <div style={{ display: 'flex', alignItems: 'center', position: 'relative', minWidth: 220 }}>
          <Search
            size={14}
            style={{
              position: 'absolute',
              left: 10,
              color: 'var(--color-muted)',
              pointerEvents: 'none',
            }}
          />
          <input
            type="text"
            className="input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search feature, sensor, stat…"
            style={{
              fontSize: 'var(--text-xs)',
              paddingLeft: 30,
              paddingRight: search ? 28 : 10,
              height: 32,
              width: '100%',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
            }}
            aria-label="Filter features by search term"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              style={{
                position: 'absolute',
                right: 6,
                background: 'transparent',
                border: 'none',
                color: 'var(--color-muted)',
                cursor: 'pointer',
                padding: 2,
              }}
              aria-label="Clear search"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Main Feature Importance Grid: Chart + Inspector */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(320px, 1.4fr) minmax(300px, 1fr)',
          gap: 'var(--space-lg)',
          alignItems: 'start',
        }}
      >
        <div className="card" style={{ padding: 'var(--space-md)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-sm)' }}>
            <h3 style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Ranked Feature Importance ({metric === 'gain' ? 'Gain %' : 'Splits'})
            </h3>
            <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
              Showing {filteredFeatures.length} of {features.length} features
            </span>
          </div>
          <FeatureImportanceChart
            features={filteredFeatures}
            selectedFeatureName={selectedFeature?.feature_name ?? null}
            onSelectFeature={(name) => setSelectedFeatureName(name)}
            metric={metric}
          />
        </div>

        <FeatureDetailInspector
          feature={selectedFeature}
          totalFeatures={features.length}
          trajectoriesManifest={trajectoriesManifest}
        />
      </div>

      {/* Family Breakdown & Correlation Comparison Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: 'var(--space-lg)',
        }}
      >
        <FeatureFamilyBreakdown dataset={dataset} familyRecords={familyRecords} />
        <ImportanceVsCorrelation
          features={features}
          onSelectFeature={(name) => {
            setSelectedFeatureName(name);
            const el = document.getElementById('feature-detail-inspector');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
          }}
        />
      </div>
    </div>
  );
};
