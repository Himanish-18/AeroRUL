import React from 'react';
import { TestMetricRecord } from '../../lib/data/types';
import { formatCycles } from '../../lib/utils/format';

interface PrimaryMetricTileProps {
  metrics: TestMetricRecord;
}

export const PrimaryMetricTile: React.FC<PrimaryMetricTileProps> = ({ metrics }) => {
  return (
    <div
      className="panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: 'var(--space-lg)',
        minHeight: '156px',
        borderColor: 'var(--color-rule-2)',
        background: 'var(--color-paper-2)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-ink)' }}>
            Endpoint MAE
          </span>
          <span
            style={{
              fontSize: '0.7rem',
              color: 'var(--color-accent)',
              border: 'var(--rule-hair) solid var(--color-accent-soft)',
              padding: '1px 6px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--color-accent-soft)',
            }}
          >
            Primary benchmark metric
          </span>
        </div>
        <span className="chip chip-cool">
          {metrics.dataset} · {metrics.selected_model}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-xs)', margin: 'var(--space-sm) 0' }}>
        <span
          className="measured"
          style={{
            fontSize: '3rem',
            fontWeight: 500,
            lineHeight: 1,
            color: 'var(--color-ink)',
            letterSpacing: '-0.04em',
          }}
        >
          {formatCycles(metrics.endpoint_mae)}
        </span>
        <span style={{ fontSize: 'var(--text-md)', color: 'var(--color-muted)' }}>
          operating cycles
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
        <span>Official test set evaluation endpoint</span>
        <span style={{ color: 'var(--color-ink-2)' }}>
          Signed error: {metrics.mean_signed_error > 0 ? '+' : ''}{metrics.mean_signed_error.toFixed(2)} cycles
        </span>
      </div>
    </div>
  );
};

