import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Cpu, Sliders, Layers, Clock } from 'lucide-react';
import { DatasetManifestConfig } from '../../lib/data/types';

interface ModelMetadataCardProps {
  manifestConfig: DatasetManifestConfig;
  frozenTimestamp: string;
}

export const ModelMetadataCard: React.FC<ModelMetadataCardProps> = ({
  manifestConfig,
  frozenTimestamp,
}) => {
  const [showAllParams, setShowAllParams] = useState(false);

  const hyperparams = manifestConfig.hyperparameters || {};
  const entries = Object.entries(hyperparams);
  const visibleEntries = showAllParams ? entries : entries.slice(0, 6);

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <Cpu size={18} color="var(--color-cool)" />
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Frozen Model Specification
          </h3>
        </div>
        <span className="chip chip-cool">
          {manifestConfig.model_family} · {manifestConfig.dataset}
        </span>
      </div>

      {/* Grid of key structural parameters */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        <div className="panel-subtle" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Layers size={12} /> Feature window
          </span>
          <span className="measured" style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
            {manifestConfig.feature_configuration}
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            {manifestConfig.n_features} causal features
          </span>
        </div>

        <div className="panel-subtle" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Sliders size={12} /> Target cap
          </span>
          <span className="measured" style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
            {manifestConfig.target_cap === 'None' ? 'None (uncapped)' : `${manifestConfig.target_cap} cycles`}
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            {manifestConfig.output_constraint}
          </span>
        </div>

        <div className="panel-subtle" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Cpu size={12} /> Inner-CV score
          </span>
          <span className="measured" style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
            {manifestConfig.inner_cv_scores?.engine_macro_mae?.toFixed(2) ?? '—'} cycles
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            3-fold engine-grouped CV
          </span>
        </div>

        <div className="panel-subtle" style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Clock size={12} /> Manifest frozen
          </span>
          <span className="measured" style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
            {frozenTimestamp.split(' ')[0]}
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            Immutable reference
          </span>
        </div>
      </div>

      {/* Hyperparameters subsection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', fontWeight: 500 }}>
            Tuned hyperparameters ({entries.length} parameters)
          </span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setShowAllParams(!showAllParams)}
            style={{ padding: '0 var(--space-xs)', height: '24px', fontSize: 'var(--text-xs)' }}
          >
            {showAllParams ? (
              <>
                <span>Show fewer</span>
                <ChevronUp size={12} />
              </>
            ) : (
              <>
                <span>Show all</span>
                <ChevronDown size={12} />
              </>
            )}
          </button>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
            gap: 'var(--space-xs)',
            background: 'var(--color-paper-3)',
            padding: 'var(--space-sm)',
            borderRadius: 'var(--radius-pill)',
            border: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          {visibleEntries.map(([k, v]) => {
            const displayVal = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v);
            return (
              <div key={k} style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <span
                  style={{
                    fontSize: '0.7rem',
                    color: 'var(--color-muted)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={k}
                >
                  {k}
                </span>
                <span
                  className="measured"
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-ink-2)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={String(v)}
                >
                  {displayVal}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
