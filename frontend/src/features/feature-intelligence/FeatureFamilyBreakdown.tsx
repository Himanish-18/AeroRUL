import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { DatasetId, FeatureFamilyRecord } from '../../lib/data/types';
import { getFamilyBreakdownList } from '../../lib/data/featureIntelligenceAdapter';

interface FeatureFamilyBreakdownProps {
  dataset: DatasetId;
  familyRecords: FeatureFamilyRecord[];
}

export const FeatureFamilyBreakdown: React.FC<FeatureFamilyBreakdownProps> = ({
  dataset,
  familyRecords,
}) => {
  const [familyType, setFamilyType] = useState<'sensor' | 'statistic' | 'window'>('sensor');

  const items = getFamilyBreakdownList(familyRecords, dataset, familyType);

  // Take top 10 for bar chart
  const chartData = [...items].slice(0, 10).reverse().map((item) => ({
    key: item.family_key,
    gainPct: item.gain_importance_pct,
    splitPct: item.split_importance_pct,
    count: item.feature_count,
  }));

  const typeLabels = {
    sensor: 'Sensor telemetry groups',
    statistic: 'Feature statistics (mean, std, slope, r2, ...)',
    window: 'Temporal window spans & lags (W10..W75, Lag 1..5)',
  };

  return (
    <div
      className="card"
      style={{ padding: 'var(--space-lg)' }}
      data-testid="feature-family-breakdown"
      aria-label="Feature family breakdown"
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
            Feature Family Aggregations
          </h3>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 'var(--space-xxs) 0 0 0' }}>
            {typeLabels[familyType]}
          </p>
        </div>

        {/* Family Type Switcher */}
        <div style={{ display: 'flex', gap: 'var(--space-xxs)', background: 'var(--color-paper-3)', padding: 2, borderRadius: 'var(--radius-sm)' }}>
          {(['sensor', 'statistic', 'window'] as const).map((t) => (
            <button
              key={t}
              type="button"
              className={`btn btn-sm ${familyType === t ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setFamilyType(t)}
              style={{
                fontSize: 'var(--text-xs)',
                padding: '3px 10px',
                textTransform: 'capitalize',
              }}
            >
              By {t}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div style={{ width: '100%', height: 260, marginBottom: 'var(--space-md)' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" horizontal={false} />
            <XAxis
              type="number"
              unit="%"
              tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--color-rule)' }}
            />
            <YAxis
              type="category"
              dataKey="key"
              width={90}
              tick={{ fill: 'var(--color-ink)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'var(--color-rule)' }}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div
                    style={{
                      background: 'var(--color-paper-2)',
                      border: 'var(--rule-hair) solid var(--color-rule)',
                      padding: 'var(--space-xs) var(--space-sm)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 'var(--text-xs)',
                    }}
                  >
                    <div style={{ fontWeight: 600, color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>
                      {d.key}
                    </div>
                    <div style={{ color: 'var(--color-accent)' }}>
                      Gain Share: {d.gainPct.toFixed(2)}%
                    </div>
                    <div style={{ color: 'var(--color-cool)' }}>
                      Split Share: {d.splitPct.toFixed(2)}%
                    </div>
                    <div style={{ color: 'var(--color-muted)' }}>
                      Features: {d.count}
                    </div>
                  </div>
                );
              }}
            />
            <Bar dataKey="gainPct" fill="var(--color-accent)" radius={[0, 3, 3, 0]} name="Gain Share (%)" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Family Table */}
      <div style={{ overflowX: 'auto' }}>
        <table className="table" style={{ width: '100%', fontSize: 'var(--text-xs)' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Rank</th>
              <th style={{ textAlign: 'left' }}>Family Key</th>
              <th style={{ textAlign: 'right' }}>Features</th>
              <th style={{ textAlign: 'right' }}>Total Gain (%)</th>
              <th style={{ textAlign: 'right' }}>Total Splits (%)</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.family_key}>
                <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-muted)' }}>
                  #{item.rank_in_family}
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--color-ink)' }}>
                  {item.family_key}
                </td>
                <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {item.feature_count}
                </td>
                <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--color-accent)' }}>
                  {item.gain_importance_pct.toFixed(2)}%
                </td>
                <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--color-cool)' }}>
                  {item.split_importance_pct.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
