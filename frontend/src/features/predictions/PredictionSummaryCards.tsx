import React from 'react';
import { PredictionSummaryStats, TestMetricRecord } from '../../lib/data/types';
import { formatCycles, formatInteger, formatPercent } from '../../lib/utils/format';
import { Activity, AlertTriangle, ArrowUpRight, CheckCircle2 } from 'lucide-react';

interface PredictionSummaryCardsProps {
  summary: PredictionSummaryStats;
  metric?: TestMetricRecord;
}

export const PredictionSummaryCards: React.FC<PredictionSummaryCardsProps> = ({
  summary,
  metric,
}) => {
  const totalOfficial = metric?.n_official_test_engines ?? summary.totalEvaluated;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: 'var(--space-md)',
      }}
    >
      {/* 1. Engines Evaluated */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Engines Evaluated
          </span>
          <CheckCircle2 size={16} color="var(--color-cool)" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-xs)' }}>
          <span className="measured" style={{ fontSize: '1.75rem', fontWeight: 500, color: 'var(--color-ink)', lineHeight: 1.1 }}>
            {formatInteger(summary.totalEvaluated)}
          </span>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            / {formatInteger(totalOfficial)} total
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', marginTop: 'auto' }}>
          <span className="chip chip-cool" style={{ fontSize: '0.7rem' }}>
            {summary.coveragePercent}% coverage
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            Zero test engines censored or omitted
          </span>
        </div>
      </div>

      {/* 2. Overpredictions */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Overpredictions
          </span>
          <ArrowUpRight size={16} color="var(--color-ok)" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-xs)' }}>
          <span className="measured" style={{ fontSize: '1.75rem', fontWeight: 500, color: 'var(--color-ink)', lineHeight: 1.1 }}>
            {formatInteger(summary.overpredictionCount)}
          </span>
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-muted)' }}>
            ({formatPercent(summary.overpredictionPercent / 100)})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', marginTop: 'auto' }}>
          <span className="chip chip-ok" style={{ fontSize: '0.7rem' }}>
            pred &gt; actual
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            Premature maintenance inspection
          </span>
        </div>
      </div>

      {/* 3. Underpredictions */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Underpredictions
          </span>
          <AlertTriangle size={16} color="var(--color-bad)" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-xs)' }}>
          <span className="measured" style={{ fontSize: '1.75rem', fontWeight: 500, color: 'var(--color-ink)', lineHeight: 1.1 }}>
            {formatInteger(summary.underpredictionCount)}
          </span>
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-muted)' }}>
            ({formatPercent(summary.underpredictionPercent / 100)})
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', marginTop: 'auto' }}>
          <span className="chip chip-bad" style={{ fontSize: '0.7rem' }}>
            pred &lt; actual
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            Late servicing risk (penalized heavily)
          </span>
        </div>
      </div>

      {/* 4. Endpoint MAE & Dispersion */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Endpoint MAE & Dispersion
          </span>
          <Activity size={16} color="var(--color-accent)" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-xs)' }}>
          <span className="measured" style={{ fontSize: '1.75rem', fontWeight: 500, color: 'var(--color-ink)', lineHeight: 1.1 }}>
            {formatCycles(summary.endpointMae)}
          </span>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>cycles MAE</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', marginTop: 'auto', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            Median: <strong className="measured" style={{ color: 'var(--color-ink-2)' }}>{formatCycles(summary.medianAbsoluteError)}</strong>
          </span>
          <span style={{ color: 'var(--color-rule-2)' }}>·</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            P90: <strong className="measured" style={{ color: 'var(--color-ink-2)' }}>{formatCycles(summary.p90AbsoluteError)}</strong> cycles
          </span>
        </div>
      </div>
    </div>
  );
};
