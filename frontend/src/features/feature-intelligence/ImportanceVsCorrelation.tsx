import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import { FeatureImportanceRecord } from '../../lib/data/types';
import {
  getCorrelationComparisonList,
  QuadrantType,
} from '../../lib/data/featureIntelligenceAdapter';
import { Info } from 'lucide-react';

interface ImportanceVsCorrelationProps {
  features: FeatureImportanceRecord[];
  onSelectFeature?: (featureName: string) => void;
}

const QUADRANT_COLORS: Record<QuadrantType, string> = {
  high_importance_high_corr: '#f59e0b', // Amber (High Gain / High Corr)
  high_importance_low_corr: '#a855f7',  // Purple (High Gain / Low Corr)
  low_importance_high_corr: '#38bdf8',  // Cyan (Low Gain / High Corr)
  low_importance_low_corr: '#64748b',   // Slate (Low Gain / Low Corr)
};

export const ImportanceVsCorrelation: React.FC<ImportanceVsCorrelationProps> = ({
  features,
  onSelectFeature,
}) => {
  const [corrMetric, setCorrMetric] = useState<'pearson' | 'spearman'>('pearson');
  const [selectedQuadrant, setSelectedQuadrant] = useState<QuadrantType | 'all'>('all');

  // Metric-specific comparison: never fallback to alternate metric or 0
  const comparisonItems = getCorrelationComparisonList(features, corrMetric);

  const filteredItems = selectedQuadrant === 'all'
    ? comparisonItems
    : comparisonItems.filter((item) => item.quadrant === selectedQuadrant);

  const scatterData = filteredItems.map((item) => {
    const rawVal = corrMetric === 'pearson' ? item.correlation_pearson : item.correlation_spearman;
    const absVal = corrMetric === 'pearson' ? item.abs_pearson : item.abs_spearman;
    return {
      name: item.feature_name,
      sensor: item.sensor,
      stat: item.statistic,
      gainPct: item.gain_importance_pct,
      corr: absVal!, // Strictly defined for metric
      rawCorr: rawVal,
      quadrant: item.quadrant,
      quadrantLabel: item.quadrantLabel,
    };
  });

  const metricName = corrMetric === 'pearson' ? 'Absolute Pearson |r|' : 'Absolute Spearman |ρ|';

  return (
    <div
      className="card"
      style={{ padding: 'var(--space-lg)' }}
      data-testid="importance-vs-correlation"
      aria-label="Importance versus correlation comparison"
    >
      {/* Header & Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
            Model Gain Importance vs. Target Correlation
          </h3>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 'var(--space-xxs) 0 0 0' }}>
            Comparing multivariate tree gain objective against univariate linear/monotonic target association ({comparisonItems.length} evaluated features)
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
          {/* Metric Selector */}
          <div style={{ display: 'flex', background: 'var(--color-paper-3)', padding: 2, borderRadius: 'var(--radius-sm)' }}>
            <button
              type="button"
              className={`btn btn-sm ${corrMetric === 'pearson' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setCorrMetric('pearson');
                setSelectedQuadrant('all');
              }}
              style={{ fontSize: 'var(--text-xs)', padding: '3px 8px' }}
              aria-label="Select Pearson correlation metric"
              data-testid="btn-metric-pearson"
            >
              Pearson |r|
            </button>
            <button
              type="button"
              className={`btn btn-sm ${corrMetric === 'spearman' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setCorrMetric('spearman');
                setSelectedQuadrant('all');
              }}
              style={{ fontSize: 'var(--text-xs)', padding: '3px 8px' }}
              aria-label="Select Spearman correlation metric"
              data-testid="btn-metric-spearman"
            >
              Spearman |ρ|
            </button>
          </div>

          {/* Quadrant Filter */}
          <select
            value={selectedQuadrant}
            onChange={(e) => setSelectedQuadrant(e.target.value as QuadrantType | 'all')}
            style={{
              fontSize: 'var(--text-xs)',
              background: 'var(--color-paper-3)',
              color: 'var(--color-ink)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              borderRadius: 'var(--radius-sm)',
              padding: '4px 8px',
            }}
            aria-label="Filter features by quadrant"
            data-testid="select-quadrant-filter"
          >
            <option value="all">All Quadrants ({comparisonItems.length})</option>
            <option value="high_importance_high_corr">High Gain / High Correlation</option>
            <option value="high_importance_low_corr">High Gain / Low Correlation</option>
            <option value="low_importance_high_corr">Low Gain / High Correlation</option>
            <option value="low_importance_low_corr">Low Gain / Low Correlation</option>
          </select>
        </div>
      </div>

      {/* Legend Chips */}
      <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap', marginBottom: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: QUADRANT_COLORS.high_importance_high_corr }} />
          <span style={{ color: 'var(--color-muted)' }}>High Gain / High Correlation</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: QUADRANT_COLORS.high_importance_low_corr }} />
          <span style={{ color: 'var(--color-muted)' }}>High Gain / Low Correlation</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: QUADRANT_COLORS.low_importance_high_corr }} />
          <span style={{ color: 'var(--color-muted)' }}>Low Gain / High Correlation</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: QUADRANT_COLORS.low_importance_low_corr }} />
          <span style={{ color: 'var(--color-muted)' }}>Low Gain / Low Correlation</span>
        </div>
      </div>

      {/* Scatter Chart */}
      <div style={{ width: '100%', height: 320, marginBottom: 'var(--space-md)' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
            <XAxis
              type="number"
              dataKey="corr"
              name={metricName}
              domain={[0, 1]}
              tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--color-rule)' }}
              label={{
                value: metricName,
                position: 'bottom',
                offset: 5,
                fill: 'var(--color-muted)',
                fontSize: 11,
              }}
            />
            <YAxis
              type="number"
              dataKey="gainPct"
              name="Gain Importance (%)"
              unit="%"
              tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--color-rule)' }}
              label={{
                value: 'Model Gain Share (%)',
                angle: -90,
                position: 'left',
                offset: -5,
                fill: 'var(--color-muted)',
                fontSize: 11,
              }}
            />
            <ZAxis range={[50, 50]} />
            <ReferenceLine x={0.5} stroke="var(--color-rule)" strokeDasharray="4 4" />
            <ReferenceLine y={2.0} stroke="var(--color-rule)" strokeDasharray="4 4" />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                const rawFormatted = (typeof d.rawCorr === 'number')
                  ? (d.rawCorr > 0 ? `+${d.rawCorr.toFixed(4)}` : d.rawCorr.toFixed(4))
                  : '—';
                const corrFormatted = (typeof d.corr === 'number') ? d.corr.toFixed(4) : '—';
                return (
                  <div
                    style={{
                      background: 'var(--color-paper-2)',
                      border: 'var(--rule-hair) solid var(--color-rule)',
                      padding: 'var(--space-xs) var(--space-sm)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 'var(--text-xs)',
                      boxShadow: 'var(--shadow-md)',
                      maxWidth: 260,
                    }}
                  >
                    <div style={{ fontWeight: 600, color: 'var(--color-ink)', fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>
                      {d.name}
                    </div>
                    <div style={{ color: 'var(--color-accent)', marginTop: 2 }}>
                      Gain Share: {typeof d.gainPct === 'number' ? d.gainPct.toFixed(2) : d.gainPct}%
                    </div>
                    <div style={{ color: '#38bdf8' }}>
                      {metricName}: {corrFormatted} (raw: {rawFormatted})
                    </div>
                    <div style={{ fontSize: '0.65rem', color: QUADRANT_COLORS[d.quadrant as QuadrantType], marginTop: 4 }}>
                      {d.quadrantLabel}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter
              data={scatterData}
              cursor="pointer"
              onClick={(pt) => onSelectFeature && onSelectFeature(pt.name)}
            >
              {scatterData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={QUADRANT_COLORS[entry.quadrant as QuadrantType]}
                  opacity={0.85}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Methodological Context Box */}
      <div
        style={{
          background: 'var(--color-paper-1)',
          border: 'var(--rule-hair) solid var(--color-rule)',
          borderRadius: 'var(--radius-sm)',
          padding: 'var(--space-sm) var(--space-md)',
          fontSize: 'var(--text-xs)',
          color: 'var(--color-muted)',
          display: 'flex',
          gap: 'var(--space-sm)',
          alignItems: 'flex-start',
        }}
      >
        <Info size={16} style={{ color: 'var(--color-accent)', flexShrink: 0, marginTop: 1 }} />
        <div>
          <strong style={{ color: 'var(--color-ink)' }}>Understanding Model Gain vs. Target Correlation:</strong>
          <span style={{ marginLeft: 4 }}>
            Univariate correlation measures linear (Pearson) or monotonic (Spearman) association with the target in isolation.
            LightGBM builds an ensemble of gradient-boosted trees that evaluate features jointly.
            Features in the High Gain / Low Correlation quadrant may indicate nonlinear decision thresholds or interaction effects, though these are possible explanations rather than established causal mechanisms.
            Conversely, features in the Low Gain / High Correlation quadrant may reflect shared or redundant signal with higher-gain features, diluting split utility. Correlation does not imply physical causality.
          </span>
        </div>
      </div>
    </div>
  );
};
