import React, { useEffect } from 'react';
import { X, Info } from 'lucide-react';
import { PredictionRecord } from '../../lib/data/types';
import { formatCycles, formatSignedError, formatSignedDelta } from '../../lib/utils/format';

interface EngineEndpointInspectorProps {
  prediction: PredictionRecord | null;
  onClose: () => void;
}

export const EngineEndpointInspector: React.FC<EngineEndpointInspectorProps> = ({
  prediction,
  onClose,
}) => {
  // Support Escape key to close inspector
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!prediction) return null;

  const isOver = prediction.prediction_error > 0;
  const isExact = Math.abs(prediction.prediction_error) < 0.001;
  const isUnder = prediction.prediction_error < 0;
  const relError = prediction.actual_rul_official > 0
    ? (prediction.prediction_error / prediction.actual_rul_official) * 100
    : 0;

  const postProcessingDelta = prediction.predicted_rul_final - prediction.predicted_rul_raw;
  const hasPostProcessingChange = Math.abs(postProcessingDelta) > 0.01;

  return (
    <div
      className="panel"
      role="region"
      aria-label={`Engine endpoint inspection for unit ${prediction.unit_id}`}
      style={{
        border: 'var(--rule-hair) solid var(--color-rule-2)',
        background: 'var(--color-paper-3)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-md)',
        padding: 'var(--space-md) var(--space-lg)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-md)', fontWeight: 500, color: 'var(--color-ink)' }}>
            Engine Endpoint Detail: Unit #{prediction.unit_id}
          </span>
          <span className="chip chip-cool">{prediction.dataset}</span>
          <span className={`chip ${isOver ? 'chip-accent' : isExact ? '' : 'chip-cool'}`}>
            {isOver
              ? 'Overprediction (optimistic)'
              : isExact
              ? 'Exact parity'
              : 'Underprediction (conservative)'}
          </span>
        </div>

        <button
          type="button"
          className="btn btn-icon btn-sm"
          onClick={onClose}
          aria-label="Close inspector drawer"
        >
          <X size={16} />
        </button>
      </div>

      {/* Primary Metrics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        {/* 1. Censoring Endpoint */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Last observed cycle</span>
          <div className="measured" style={{ fontSize: '1.25rem', fontWeight: 500, color: 'var(--color-ink)', marginTop: '2px' }}>
            {prediction.last_observed_cycle}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Censoring endpoint</span>
        </div>

        {/* 2. NASA Ground Truth */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Actual RUL (official)</span>
          <div className="measured" style={{ fontSize: '1.25rem', fontWeight: 500, color: 'var(--color-ink)', marginTop: '2px' }}>
            {formatCycles(prediction.actual_rul_official)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Ground truth cycles</span>
        </div>

        {/* 3. Raw Model Output */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Raw predicted RUL</span>
          <div className="measured" style={{ fontSize: '1.25rem', fontWeight: 500, color: 'var(--color-ink-2)', marginTop: '2px' }}>
            {formatCycles(prediction.predicted_rul_raw)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Pre-constraint output</span>
        </div>

        {/* 4. Final Post-Processed Output */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Final predicted RUL</span>
          <div className="measured" style={{ fontSize: '1.25rem', fontWeight: 500, color: 'var(--color-ink)', marginTop: '2px' }}>
            {formatCycles(prediction.predicted_rul_final)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Bounded benchmark output</span>
        </div>

        {/* 5. Signed Error */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Signed error</span>
          <div
            className="measured"
            style={{
              fontSize: '1.25rem',
              fontWeight: 500,
              color: isOver ? 'var(--color-accent)' : isUnder ? 'var(--color-cool)' : 'var(--color-ink)',
              marginTop: '2px',
            }}
          >
            {formatSignedError(prediction.prediction_error)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            {isOver ? '+ over (optimistic)' : isUnder ? '- under (conservative)' : 'exact'}
          </span>
        </div>

        {/* 6. Absolute & Relative Error */}
        <div className="panel" style={{ padding: 'var(--space-sm)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Absolute error</span>
          <div className="measured" style={{ fontSize: '1.25rem', fontWeight: 500, color: 'var(--color-ink-2)', marginTop: '2px' }}>
            {formatCycles(prediction.absolute_error)}
          </div>
          <span style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
            Relative: {Math.abs(relError).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Post-processing & Safety Decomposition Note */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        {/* Post-processing constraint card */}
        <div
          className="panel"
          style={{
            background: 'var(--color-paper-2)',
            padding: 'var(--space-sm) var(--space-md)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--space-xs)',
          }}
        >
          <Info size={16} color="var(--color-cool)" style={{ marginTop: '2px', flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: 'var(--text-xs)' }}>
            <span style={{ fontWeight: 500, color: 'var(--color-ink)' }}>
              Post-processing constraint difference: {formatSignedDelta(postProcessingDelta)} cycles
            </span>
            <span style={{ color: 'var(--color-muted)' }}>
              {hasPostProcessingChange
                ? `Model output adjusted from raw ${formatCycles(prediction.predicted_rul_raw)} to ${formatCycles(prediction.predicted_rul_final)} via target cap (${prediction.selected_target_cap} cycles) / non-negative bounds.`
                : `Unconstrained output: Raw model output is identical to final evaluated prediction (no target cap clipping was active for this endpoint).`}
            </span>
          </div>
        </div>

        {/* Operational safety impact card */}
        <div
          className="panel"
          style={{
            background: 'var(--color-paper-2)',
            padding: 'var(--space-sm) var(--space-md)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--space-xs)',
          }}
        >
          <Info size={16} color={isOver ? 'var(--color-accent)' : 'var(--color-cool)'} style={{ marginTop: '2px', flexShrink: 0 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: 'var(--text-xs)' }}>
            <span style={{ fontWeight: 500, color: 'var(--color-ink)' }}>
              Operational Domain Context
            </span>
            <span style={{ color: 'var(--color-muted)' }}>
              {isOver
                ? `Engine RUL was overpredicted by ${formatCycles(prediction.prediction_error)} cycles (optimistic estimate: predicted ${formatCycles(prediction.predicted_rul_final)} vs actual ${formatCycles(prediction.actual_rul_official)}). If used directly in a threshold-based maintenance policy, this could delay inspection or servicing.`
                : isExact
                ? `Exact parity with official test ground truth. Zero prediction bias observed for this endpoint.`
                : `Engine RUL was underpredicted by ${formatCycles(Math.abs(prediction.prediction_error))} cycles (conservative estimate: predicted ${formatCycles(prediction.predicted_rul_final)} vs actual ${formatCycles(prediction.actual_rul_official)}). Conservative estimates could trigger maintenance inspection or servicing earlier than necessary.`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
