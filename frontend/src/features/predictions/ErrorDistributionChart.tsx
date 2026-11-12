import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
} from 'recharts';
import { ErrorDistributionBin, PredictionRecord } from '../../lib/data/types';
import { getErrorDistribution } from '../../lib/data/adapter';

interface ErrorDistributionChartProps {
  predictions: PredictionRecord[];
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: ErrorDistributionBin;
  }>;
}

const DistributionTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;

  let categoryLabel = 'Balanced / Near-Zero';
  let badgeClass = 'chip';
  if (data.isUnderprediction) {
    categoryLabel = 'Underpredicted (Conservative / High Risk)';
    badgeClass = 'chip chip-bad';
  } else if (data.isOverprediction) {
    categoryLabel = 'Overpredicted (Premature Overhaul)';
    badgeClass = 'chip chip-ok';
  }

  return (
    <div
      style={{
        background: 'var(--color-paper-3)',
        border: 'var(--rule-hair) solid var(--color-rule-2)',
        borderRadius: 'var(--radius-pill)',
        padding: 'var(--space-sm) var(--space-md)',
        boxShadow: 'var(--shadow-tooltip)',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        fontSize: 'var(--text-xs)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-md)' }}>
        <span style={{ fontWeight: 500, color: 'var(--color-ink)' }}>
          Bin: {data.binLabel} cycles
        </span>
        <span className={badgeClass}>{categoryLabel}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '2px 12px', marginTop: '2px' }}>
        <span style={{ color: 'var(--color-muted)' }}>Engine Count:</span>
        <span className="measured" style={{ color: 'var(--color-ink)', textAlign: 'right' }}>
          {data.count} engines
        </span>

        <span style={{ color: 'var(--color-muted)' }}>Fleet Share:</span>
        <span className="measured" style={{ color: 'var(--color-ink)', textAlign: 'right' }}>
          {data.percentage.toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

export const ErrorDistributionChart: React.FC<ErrorDistributionChartProps> = ({ predictions }) => {
  const bins = useMemo(() => getErrorDistribution(predictions, 10), [predictions]);

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Title & Legend */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Prediction Error Distribution
          </h3>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Signed error frequency (bins = 10 cycles) · Zero reference line separates under- vs over-predictions
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '2px',
                background: 'var(--color-bad)',
                display: 'inline-block',
              }}
            />
            <span style={{ color: 'var(--color-muted)' }}>Underprediction (&lt; 0)</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '2px',
                background: 'var(--color-ok)',
                display: 'inline-block',
              }}
            />
            <span style={{ color: 'var(--color-muted)' }}>Overprediction (&gt; 0)</span>
          </div>
        </div>
      </div>

      {/* Chart container */}
      <div style={{ width: '100%', height: '340px', position: 'relative' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 16, right: 24, bottom: 28, left: 16 }}>
            <CartesianGrid strokeDasharray="0" stroke="var(--color-rule)" />

            <XAxis
              dataKey="binLabel"
              tick={{ fill: 'var(--color-muted)', fontSize: 11, fontFamily: 'var(--font-outlier)' }}
              stroke="var(--color-rule-2)"
              interval="preserveStartEnd"
              label={{
                value: 'Signed error range (cycles)',
                position: 'insideBottom',
                offset: -14,
                fill: 'var(--color-muted)',
                fontSize: 12,
              }}
            />

            <YAxis
              tick={{ fill: 'var(--color-muted)', fontSize: 11, fontFamily: 'var(--font-outlier)' }}
              stroke="var(--color-rule-2)"
              width={45}
              label={{
                value: 'Engines (count)',
                angle: -90,
                position: 'insideLeft',
                offset: 10,
                style: { textAnchor: 'middle' },
                fill: 'var(--color-muted)',
                fontSize: 12,
              }}
            />

            <Tooltip content={<DistributionTooltip />} cursor={{ fill: 'var(--color-paper-3)', opacity: 0.5 }} />

            <ReferenceLine y={0} stroke="var(--color-rule-2)" />

            <Bar dataKey="count" radius={[2, 2, 0, 0]}>
              {bins.map((entry, index) => {
                let cellColor = 'var(--color-ink-2)';
                if (entry.isUnderprediction) {
                  cellColor = 'var(--color-bad)';
                } else if (entry.isOverprediction) {
                  cellColor = 'var(--color-ok)';
                }
                return <Cell key={`cell-${index}`} fill={cellColor} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
