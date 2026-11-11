import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { PredictionRecord } from '../../lib/data/types';
import { getSharedParityDomain } from '../../lib/data/adapter';
import { formatCycles, formatSignedError } from '../../lib/utils/format';

interface ActualVsPredictedScatterProps {
  predictions: PredictionRecord[];
  selectedUnitId: number | null;
  onSelectEngine: (unitId: number) => void;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: PredictionRecord;
  }>;
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;

  const isOver = data.prediction_error > 0;

  return (
    <div
      style={{
        background: 'var(--color-paper-3)',
        border: 'var(--rule-hair) solid var(--color-rule-2)',
        borderRadius: 'var(--radius-pill)',
        padding: 'var(--space-sm) var(--space-md)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.4)',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        fontSize: 'var(--text-xs)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-md)' }}>
        <span style={{ fontWeight: 500, color: 'var(--color-ink)' }}>
          Engine Unit #{data.unit_id}
        </span>
        <span className={`chip ${isOver ? 'chip-ok' : 'chip-bad'}`}>
          {isOver ? 'Overprediction' : 'Underprediction'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '2px 12px', marginTop: '2px' }}>
        <span style={{ color: 'var(--color-muted)' }}>Actual RUL:</span>
        <span className="measured" style={{ color: 'var(--color-ink)', textAlign: 'right' }}>
          {formatCycles(data.actual_rul_official)} cycles
        </span>

        <span style={{ color: 'var(--color-muted)' }}>Predicted RUL:</span>
        <span className="measured" style={{ color: 'var(--color-ink)', textAlign: 'right' }}>
          {formatCycles(data.predicted_rul_final)} cycles
        </span>

        <span style={{ color: 'var(--color-muted)' }}>Signed error:</span>
        <span className="measured" style={{ color: 'var(--color-ink-2)', textAlign: 'right' }}>
          {formatSignedError(data.prediction_error)} cycles
        </span>

        <span style={{ color: 'var(--color-muted)' }}>Absolute error:</span>
        <span className="measured" style={{ color: 'var(--color-ink-2)', textAlign: 'right' }}>
          {formatCycles(data.absolute_error)} cycles
        </span>
      </div>
    </div>
  );
};

export const ActualVsPredictedScatter: React.FC<ActualVsPredictedScatterProps> = ({
  predictions,
  selectedUnitId,
  onSelectEngine,
}) => {
  const domain = useMemo(() => getSharedParityDomain(predictions), [predictions]);

  // Format data for Recharts scatter
  const chartData = useMemo(() => {
    return predictions.map((p) => ({
      ...p,
      x: p.actual_rul_official,
      y: p.predicted_rul_final,
    }));
  }, [predictions]);

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Title & Legend */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Actual vs. Predicted RUL
          </h3>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Official test engines (n = {predictions.length}) · Deviations from the diagonal line represent error
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: 'var(--color-accent)',
                display: 'inline-block',
              }}
            />
            <span style={{ color: 'var(--color-ink-2)' }}>Engine endpoint</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '16px',
                height: '2px',
                background: 'var(--color-cool)',
                display: 'inline-block',
              }}
            />
            <span style={{ color: 'var(--color-muted)' }}>Ideal parity (y = x)</span>
          </div>
        </div>
      </div>

      {/* Chart container */}
      <div style={{ width: '100%', height: '380px', position: 'relative' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 16, right: 24, bottom: 24, left: 16 }}
            onClick={(e) => {
              if (e && e.activePayload && e.activePayload.length > 0) {
                const rec = e.activePayload[0].payload as PredictionRecord;
                if (rec && rec.unit_id) {
                  onSelectEngine(rec.unit_id);
                }
              }
            }}
          >
            <CartesianGrid strokeDasharray="0" stroke="var(--color-rule)" />

            <XAxis
              type="number"
              dataKey="x"
              name="Actual RUL"
              unit=" cycles"
              domain={domain}
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-outlier)' }}
              stroke="var(--color-rule-2)"
              label={{
                value: 'Actual RUL (official ground truth cycles)',
                position: 'insideBottom',
                offset: -12,
                fill: 'var(--color-muted)',
                fontSize: 12,
              }}
            />

            <YAxis
              type="number"
              dataKey="y"
              name="Predicted RUL"
              unit=" cycles"
              domain={domain}
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-outlier)' }}
              stroke="var(--color-rule-2)"
              label={{
                value: 'Predicted RUL (LightGBM cycles)',
                angle: -90,
                position: 'insideLeft',
                offset: 4,
                fill: 'var(--color-muted)',
                fontSize: 12,
              }}
            />

            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'var(--color-rule-2)' }} />

            {/* Ideal parity reference line representing y = x */}
            <ReferenceLine
              segment={[
                { x: domain[0], y: domain[0] },
                { x: domain[1], y: domain[1] },
              ]}
              stroke="var(--color-cool)"
              strokeWidth={1.5}
            />

            <Scatter
              name="Engines"
              data={chartData}
              fill="var(--color-accent)"
              shape={(props: any) => {
                const { cx, cy, payload } = props;
                const isSelected = payload.unit_id === selectedUnitId;
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 6 : 4}
                    fill={isSelected ? 'var(--color-accent)' : 'var(--color-accent)'}
                    stroke={isSelected ? 'var(--color-ink)' : 'var(--color-paper)'}
                    strokeWidth={isSelected ? 2 : 1}
                    style={{
                      cursor: 'pointer',
                      transition: 'r var(--dur-micro) var(--ease-out)',
                    }}
                  />
                );
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
