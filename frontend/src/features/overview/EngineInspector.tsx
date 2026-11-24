import React from 'react';
import { X } from 'lucide-react';
import { PredictionRecord } from '../../lib/data/types';
import { formatCycles, formatSignedError, formatPercent } from '../../lib/utils/format';

interface EngineInspectorProps {
  prediction: PredictionRecord | null;
  onClose: () => void;
}

export const EngineInspector: React.FC<EngineInspectorProps> = ({
  prediction,
  onClose,
}) => {
  if (!prediction) return null;

  const isOver = prediction.prediction_error > 0;
  const relError = prediction.actual_rul_official > 0
    ? prediction.prediction_error / prediction.actual_rul_official
    : 0;

  return (
    <div
      className="panel"
      style={{
        border: 'var(--rule-hair) solid var(--color-rule-2)',
        background: 'var(--color-paper-3)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-md)',
        padding: 'var(--space-md) var(--space-lg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-ink)' }}>
            Engine Trajectory Detail: Unit #{prediction.unit_id}
          </span>
          <span className="chip chip-cool">{prediction.dataset}</span>
        </div>

        <button
          type="button"
          className="btn btn-icon btn-sm"
          onClick={onClose}
          aria-label="Close inspector"
        >
          <X size={16} />
        </button>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Last observed cycle</span>
          <div className="measured" style={{ fontSize: '1.25rem', color: 'var(--color-ink)', marginTop: '2px' }}>
            {prediction.last_observed_cycle}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Censoring endpoint</span>
        </div>

        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Actual RUL</span>
          <div className="measured" style={{ fontSize: '1.25rem', color: 'var(--color-ink)', marginTop: '2px' }}>
            {formatCycles(prediction.actual_rul_official)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>NASA ground truth</span>
        </div>

        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Predicted RUL</span>
          <div className="measured" style={{ fontSize: '1.25rem', color: 'var(--color-ink)', marginTop: '2px' }}>
            {formatCycles(prediction.predicted_rul_final)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>LightGBM final output</span>
        </div>

        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Signed error</span>
          <div
            className="measured"
            style={{
              fontSize: '1.25rem',
              color: isOver ? 'var(--color-ok)' : 'var(--color-bad)',
              marginTop: '2px',
            }}
          >
            {formatSignedError(prediction.prediction_error)}
          </div>
          <span style={{ fontSize: '0.7rem', color: isOver ? 'var(--color-ok)' : 'var(--color-bad)' }}>
            {isOver ? 'Overprediction' : 'Underprediction (costlier)'}
          </span>
        </div>

        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Absolute error</span>
          <div className="measured" style={{ fontSize: '1.25rem', color: 'var(--color-ink-2)', marginTop: '2px' }}>
            {formatCycles(prediction.absolute_error)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            Relative: {formatPercent(Math.abs(relError))}
          </span>
        </div>
      </div>
    </div>
  );
};
