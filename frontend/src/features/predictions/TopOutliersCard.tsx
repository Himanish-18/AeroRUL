import React, { useMemo } from 'react';
import { PredictionRecord } from '../../lib/data/types';
import { getTopOutliers } from '../../lib/data/adapter';
import { formatCycles, formatSignedError } from '../../lib/utils/format';
import { AlertCircle, ChevronRight } from 'lucide-react';

interface TopOutliersCardProps {
  predictions: PredictionRecord[];
  selectedUnitId: number | null;
  onSelectEngine: (unitId: number) => void;
}

export const TopOutliersCard: React.FC<TopOutliersCardProps> = ({
  predictions,
  selectedUnitId,
  onSelectEngine,
}) => {
  const outliers = useMemo(() => getTopOutliers(predictions, 5), [predictions]);

  if (outliers.length === 0) return null;

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <AlertCircle size={16} color="var(--color-accent)" />
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Largest Prediction Errors (Top 5 Fleet Outliers)
          </h3>
        </div>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
          Whole-dataset ranking · Ranked by absolute endpoint error
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 'var(--space-sm)',
        }}
      >
        {outliers.map((engine, idx) => {
          const isSelected = engine.unit_id === selectedUnitId;
          const isOver = engine.prediction_error > 0;

          return (
            <div
              key={engine.unit_id}
              className="panel"
              style={{
                background: isSelected ? 'var(--color-accent-soft)' : 'var(--color-paper-3)',
                borderColor: isSelected ? 'var(--color-accent)' : 'var(--color-rule)',
                padding: 'var(--space-sm) var(--space-md)',
                display: 'flex',
                flexDirection: 'column',
                gap: 'var(--space-2xs)',
                position: 'relative',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', fontWeight: 500 }}>
                  #{idx + 1} Outlier
                </span>
                <span className={`chip ${isOver ? 'chip-accent' : 'chip-cool'}`} style={{ fontSize: '0.65rem' }}>
                  {isOver ? '+ over (optimistic)' : '- under (conservative)'}
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginTop: '2px' }}>
                <span className="measured" style={{ fontSize: 'var(--text-md)', fontWeight: 500, color: 'var(--color-ink)' }}>
                  Unit #{engine.unit_id}
                </span>
                <span className="measured" style={{ fontSize: 'var(--text-sm)', color: isOver ? 'var(--color-accent)' : 'var(--color-cool)' }}>
                  {formatSignedError(engine.prediction_error)} cyc
                </span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                <span>Actual: <strong className="measured" style={{ color: 'var(--color-ink-2)' }}>{formatCycles(engine.actual_rul_official, 0)}</strong></span>
                <span>Pred: <strong className="measured" style={{ color: 'var(--color-ink-2)' }}>{formatCycles(engine.predicted_rul_final, 0)}</strong></span>
              </div>

              <button
                type="button"
                className="btn btn-sm"
                onClick={() => onSelectEngine(engine.unit_id)}
                aria-label={`Inspect outlier unit ${engine.unit_id}`}
                style={{
                  marginTop: 'var(--space-2xs)',
                  width: '100%',
                  justifyContent: 'center',
                  background: isSelected ? 'var(--color-accent)' : 'transparent',
                  color: isSelected ? 'var(--color-accent-ink)' : 'var(--color-ink)',
                  borderColor: isSelected ? 'var(--color-accent)' : 'var(--color-rule-2)',
                  fontSize: 'var(--text-xs)',
                  height: '28px',
                }}
              >
                <span>{isSelected ? 'Inspecting' : 'Inspect endpoint'}</span>
                <ChevronRight size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
