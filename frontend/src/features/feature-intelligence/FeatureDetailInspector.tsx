import React from 'react';
import { FeatureImportanceRecord, SensorTrajectoriesManifest } from '../../lib/data/types';
import { getSensorSpec } from '../../lib/data/featureIntelligenceAdapter';
import { Activity, BarChart2, Hash, Layers } from 'lucide-react';

interface FeatureDetailInspectorProps {
  feature: FeatureImportanceRecord | null;
  totalFeatures: number;
  trajectoriesManifest?: SensorTrajectoriesManifest | null;
}

export const FeatureDetailInspector: React.FC<FeatureDetailInspectorProps> = ({
  feature,
  totalFeatures,
  trajectoriesManifest,
}) => {
  if (!feature) {
    return (
      <div
        className="card"
        style={{
          padding: 'var(--space-xl)',
          textAlign: 'center',
          color: 'var(--color-muted)',
          fontSize: 'var(--text-sm)',
        }}
      >
        Select a feature from the chart or table to view technical parameters and correlation profiles.
      </div>
    );
  }

  const sensorSpec = getSensorSpec(feature.sensor, trajectoriesManifest);

  return (
    <div
      className="card"
      id="feature-detail-inspector"
      data-testid="feature-detail-inspector"
      aria-label="Feature detail inspector"
      style={{
        padding: 'var(--space-lg)',
        borderLeft: '3px solid var(--color-accent)',
        background: 'var(--color-paper-2)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 'var(--space-md)', flexWrap: 'wrap', marginBottom: 'var(--space-md)' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', marginBottom: 'var(--space-xxs)' }}>
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                padding: '1px 6px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--color-paper-3)',
                color: 'var(--color-accent)',
                border: 'var(--rule-hair) solid var(--color-rule)',
              }}
            >
              Gain Rank #{feature.gain_rank} of {totalFeatures}
            </span>
            <span
              style={{
                fontSize: '0.65rem',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                padding: '1px 6px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--color-paper-3)',
                color: 'var(--color-muted)',
                border: 'var(--rule-hair) solid var(--color-rule)',
              }}
            >
              Split Rank #{feature.split_rank}
            </span>
          </div>
          <h3
            style={{
              fontSize: 'var(--text-base)',
              fontFamily: 'var(--font-mono)',
              color: 'var(--color-ink)',
              margin: 0,
              wordBreak: 'break-all',
            }}
          >
            {feature.feature_name}
          </h3>
          <p
            style={{
              fontSize: 'var(--text-xs)',
              color: 'var(--color-muted)',
              margin: 'var(--space-xxs) 0 0 0',
            }}
          >
            {feature.description}
          </p>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: 'var(--space-sm)',
          marginBottom: 'var(--space-md)',
        }}
      >
        <div
          style={{
            background: 'var(--color-paper-1)',
            padding: 'var(--space-sm) var(--space-md)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <BarChart2 size={13} style={{ color: 'var(--color-accent)' }} />
            Gain Importance
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-accent)',
              marginTop: 2,
            }}
          >
            {feature.gain_importance_pct.toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)', fontFamily: 'var(--font-mono)' }}>
            Raw: {feature.gain_importance.toLocaleString(undefined, { maximumFractionDigits: 1 })}
          </div>
        </div>

        <div
          style={{
            background: 'var(--color-paper-1)',
            padding: 'var(--space-sm) var(--space-md)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Hash size={13} style={{ color: 'var(--color-cool)' }} />
            Split Importance
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: 'var(--color-ink)',
              marginTop: 2,
            }}
          >
            {feature.split_importance_pct.toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)', fontFamily: 'var(--font-mono)' }}>
            Tree splits: {feature.split_importance.toLocaleString()}
          </div>
        </div>

        <div
          style={{
            background: 'var(--color-paper-1)',
            padding: 'var(--space-sm) var(--space-md)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Activity size={13} style={{ color: 'var(--color-ok)' }} />
            Pearson Correlation
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: feature.correlation_pearson !== null ? 'var(--color-ink)' : 'var(--color-muted)',
              marginTop: 2,
            }}
          >
            {feature.correlation_pearson !== null
              ? (feature.correlation_pearson > 0 ? `+${feature.correlation_pearson.toFixed(4)}` : feature.correlation_pearson.toFixed(4))
              : '—'}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)' }}>
            {feature.abs_pearson !== null
              ? `|r| = ${feature.abs_pearson.toFixed(4)}`
              : 'Not in Phase 3 screen'}
          </div>
        </div>

        <div
          style={{
            background: 'var(--color-paper-1)',
            padding: 'var(--space-sm) var(--space-md)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Layers size={13} style={{ color: 'var(--color-cool)' }} />
            Spearman Rank
          </div>
          <div
            style={{
              fontSize: 'var(--text-lg)',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              color: feature.correlation_spearman !== null ? 'var(--color-ink)' : 'var(--color-muted)',
              marginTop: 2,
            }}
          >
            {feature.correlation_spearman !== null
              ? (feature.correlation_spearman > 0 ? `+${feature.correlation_spearman.toFixed(4)}` : feature.correlation_spearman.toFixed(4))
              : '—'}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)' }}>
            {feature.abs_spearman !== null
              ? `|ρ| = ${feature.abs_spearman.toFixed(4)}`
              : 'Not in Phase 3 screen'}
          </div>
        </div>
      </div>

      {/* Sensor Physical Metadata */}
      <div
        style={{
          background: 'var(--color-paper-1)',
          padding: 'var(--space-sm) var(--space-md)',
          borderRadius: 'var(--radius-sm)',
          border: 'var(--rule-hair) solid var(--color-rule)',
          fontSize: 'var(--text-xs)',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 'var(--space-xs) var(--space-md)',
          marginBottom: 'var(--space-sm)',
        }}
      >
        <div>
          <span style={{ color: 'var(--color-muted)' }}>Underlying Sensor: </span>
          <strong style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>{feature.sensor}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--color-muted)' }}>Physical Measurement: </span>
          <span style={{ color: 'var(--color-ink)' }}>{sensorSpec.name}</span>
        </div>
        <div>
          <span style={{ color: 'var(--color-muted)' }}>Measurement Unit: </span>
          <span style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>{sensorSpec.unit} ({sensorSpec.type})</span>
        </div>
        <div>
          <span style={{ color: 'var(--color-muted)' }}>Applied Statistic: </span>
          <span style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>{feature.statistic}</span>
        </div>
        <div>
          <span style={{ color: 'var(--color-muted)' }}>Temporal Window / Lag: </span>
          <span style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>{feature.window_or_lag} cycles</span>
        </div>
      </div>

      {/* Model Interpretability Notice */}
      <div
        style={{
          fontSize: '0.7rem',
          color: 'var(--color-muted)',
          lineHeight: 1.4,
          padding: 'var(--space-xs) var(--space-sm)',
          borderLeft: '2px solid var(--color-rule)',
        }}
      >
        <strong>Technical Context:</strong> Gain measures reduction in LightGBM squared-error objective across all tree splits using this feature. Split importance reflects split frequency. Neither measure establishes physical causality, and collinear features can share or redistribute importance across trees.
      </div>
    </div>
  );
};
