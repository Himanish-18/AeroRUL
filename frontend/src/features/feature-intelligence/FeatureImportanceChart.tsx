import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import { FeatureImportanceRecord } from '../../lib/data/types';

interface FeatureImportanceChartProps {
  features: FeatureImportanceRecord[];
  selectedFeatureName: string | null;
  onSelectFeature: (featureName: string) => void;
  metric: 'gain' | 'split';
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  features,
  selectedFeatureName,
  onSelectFeature,
  metric,
}) => {
  if (features.length === 0) {
    return (
      <div
        style={{
          height: 380,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--color-muted)',
          fontSize: 'var(--text-sm)',
        }}
      >
        No features match the current search or filter criteria.
      </div>
    );
  }

  // Reverse list for horizontal bar chart (top ranked at top)
  const chartData = [...features].reverse().map((f) => ({
    name: f.feature_name,
    displayName: f.feature_name.length > 22 ? `${f.feature_name.slice(0, 20)}…` : f.feature_name,
    fullName: f.feature_name,
    value: metric === 'gain' ? f.gain_importance : f.split_importance,
    pct: metric === 'gain' ? f.gain_importance_pct : f.split_importance_pct,
    sensor: f.sensor,
    statistic: f.statistic,
    rank: metric === 'gain' ? f.gain_rank : f.split_rank,
  }));

  const valueKey = 'value';
  const metricLabel = metric === 'gain' ? 'Gain Importance' : 'Split Importance';

  return (
    <div
      style={{ width: '100%', height: Math.max(380, chartData.length * 28 + 60) }}
      data-testid="feature-importance-chart"
      aria-label={`${metricLabel} bar chart`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 10, right: 30, left: 20, bottom: 10 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-rule)"
            horizontal={false}
          />
          <XAxis
            type="number"
            tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--color-rule)' }}
            tickLine={{ stroke: 'var(--color-rule)' }}
            tickFormatter={(val: number) =>
              metric === 'gain'
                ? val >= 1e6
                  ? `${(val / 1e6).toFixed(1)}M`
                  : val.toLocaleString()
                : val.toLocaleString()
            }
          />
          <YAxis
            type="category"
            dataKey="displayName"
            width={130}
            tick={{ fill: 'var(--color-ink)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: 'var(--color-rule)' }}
            tickLine={false}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload || !payload.length) return null;
              const data = payload[0].payload;
              const isSelected = data.fullName === selectedFeatureName;
              return (
                <div
                  style={{
                    background: 'var(--color-paper-2)',
                    border: 'var(--rule-hair) solid var(--color-rule)',
                    borderRadius: 'var(--radius-sm)',
                    padding: 'var(--space-sm) var(--space-md)',
                    boxShadow: 'var(--shadow-md)',
                    fontSize: 'var(--text-xs)',
                    maxWidth: 300,
                  }}
                >
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: isSelected ? 'var(--color-accent)' : 'var(--color-ink)',
                      marginBottom: 'var(--space-xs)',
                      wordBreak: 'break-all',
                    }}
                  >
                    #{data.rank} {data.fullName}
                  </div>
                  <div style={{ color: 'var(--color-muted)', marginBottom: 2 }}>
                    Sensor: <span style={{ color: 'var(--color-ink)' }}>{data.sensor}</span>
                  </div>
                  <div style={{ color: 'var(--color-muted)', marginBottom: 2 }}>
                    Statistic: <span style={{ color: 'var(--color-ink)' }}>{data.statistic}</span>
                  </div>
                  <div style={{ color: 'var(--color-muted)', marginTop: 'var(--space-xs)' }}>
                    {metricLabel}:{' '}
                    <strong style={{ color: 'var(--color-accent)' }}>
                      {typeof data.value === 'number' ? data.value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : data.value}
                    </strong>{' '}
                    ({data.pct.toFixed(2)}%)
                  </div>
                </div>
              );
            }}
          />
          <Bar
            dataKey={valueKey}
            radius={[0, 3, 3, 0]}
            onClick={(entry) => onSelectFeature(entry.fullName)}
            cursor="pointer"
          >
            {chartData.map((entry) => {
              const isSelected = entry.fullName === selectedFeatureName;
              return (
                <Cell
                  key={entry.fullName}
                  fill={isSelected ? 'var(--color-accent)' : 'var(--color-paper-3)'}
                  stroke={isSelected ? 'var(--color-accent)' : 'var(--color-rule)'}
                  strokeWidth={1}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
