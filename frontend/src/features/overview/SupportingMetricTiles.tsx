import React from 'react';
import { TestMetricRecord } from '../../lib/data/types';
import { StatTile } from '../../components/shared/StatTile';
import { formatCycles, formatR2, formatNasaScore } from '../../lib/utils/format';
import { calculateCoverage } from '../../lib/data/adapter';

interface SupportingMetricTilesProps {
  metrics: TestMetricRecord;
}

export const SupportingMetricTiles: React.FC<SupportingMetricTilesProps> = ({ metrics }) => {
  const coverage = calculateCoverage(metrics);

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: 'var(--space-md)',
      }}
    >
      <StatTile
        label="Endpoint RMSE"
        value={formatCycles(metrics.endpoint_rmse)}
        unit="cycles"
        supporting="Penalizes larger trajectory outliers"
        badge="L2 loss"
        badgeTone="cool"
      />

      <StatTile
        label="Endpoint R²"
        value={formatR2(metrics.endpoint_r2)}
        supporting="Variance explained on test set"
        badge="Fit quality"
        badgeTone="cool"
      />

      <StatTile
        label="Prediction coverage"
        value={`${coverage.coveragePercent}%`}
        unit=""
        supporting={`${coverage.successfullyPredicted} of ${coverage.totalOfficial} official test engines`}
        badge="100% complete"
        badgeTone="ok"
      />

      <StatTile
        label="NASA asymmetric score"
        value={formatNasaScore(metrics.nasa_score_sum)}
        unit="pts"
        supporting={`Mean: ${formatNasaScore(metrics.nasa_score_mean)} / engine`}
        badge="Late-penalty"
        badgeTone="cool"
      />
    </div>
  );
};
