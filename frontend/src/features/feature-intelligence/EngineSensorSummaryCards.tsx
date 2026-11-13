import React from 'react';
import { SensorTrajectoryRecord, SensorTrajectoriesManifest } from '../../lib/data/types';
import {
  getEngineSensorStatistics,
  getSensorSpec,
} from '../../lib/data/featureIntelligenceAdapter';
import { Clock, TrendingUp, Activity, BarChart, ArrowRight } from 'lucide-react';

interface EngineSensorSummaryCardsProps {
  trajectories: SensorTrajectoryRecord[];
  sensor: string;
  primaryUnitId: number;
  manifest?: SensorTrajectoriesManifest | null;
}

export const EngineSensorSummaryCards: React.FC<EngineSensorSummaryCardsProps> = ({
  trajectories,
  sensor,
  primaryUnitId,
  manifest,
}) => {
  const stats = getEngineSensorStatistics(trajectories, sensor, primaryUnitId);
  const sensorSpec = getSensorSpec(sensor, manifest);

  if (!stats) {
    return null;
  }

  const isRising = stats.netChange > 0;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: 'var(--space-sm)',
        marginBottom: 'var(--space-md)',
      }}
      data-testid="engine-sensor-summary-cards"
      aria-label="Engine sensor summary statistics"
    >
      {/* Card 1: Observed Cycles */}
      <div
        className="card"
        style={{
          padding: 'var(--space-sm) var(--space-md)',
          background: 'var(--color-paper-2)',
          border: 'var(--rule-hair) solid var(--color-rule)',
        }}
      >
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Clock size={13} style={{ color: 'var(--color-accent)' }} />
          Observed Lifetime
        </div>
        <div
          style={{
            fontSize: 'var(--text-lg)',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: 'var(--color-ink)',
            marginTop: 2,
          }}
        >
          {stats.endCycle}{' '}
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 400, color: 'var(--color-muted)' }}>cycles</span>
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)' }}>
          Unit #{primaryUnitId} run-to-failure
        </div>
      </div>

      {/* Card 2: Min & Max */}
      <div
        className="card"
        style={{
          padding: 'var(--space-sm) var(--space-md)',
          background: 'var(--color-paper-2)',
          border: 'var(--rule-hair) solid var(--color-rule)',
        }}
      >
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <BarChart size={13} style={{ color: 'var(--color-cool)' }} />
          Min / Max Range
        </div>
        <div
          style={{
            fontSize: 'var(--text-sm)',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: 'var(--color-ink)',
            marginTop: 4,
          }}
        >
          {stats.min.toFixed(2)} to {stats.max.toFixed(2)}
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)', fontFamily: 'var(--font-mono)' }}>
          Unit: {sensorSpec.unit}
        </div>
      </div>

      {/* Card 3: Mean & Std */}
      <div
        className="card"
        style={{
          padding: 'var(--space-sm) var(--space-md)',
          background: 'var(--color-paper-2)',
          border: 'var(--rule-hair) solid var(--color-rule)',
        }}
      >
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Activity size={13} style={{ color: 'var(--color-ok)' }} />
          Mean ± Std
        </div>
        <div
          style={{
            fontSize: 'var(--text-sm)',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: 'var(--color-ink)',
            marginTop: 4,
          }}
        >
          {stats.mean.toFixed(2)} ± {stats.std.toFixed(3)}
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)' }}>
          N = {stats.count} recorded points
        </div>
      </div>

      {/* Card 4: Net Drift */}
      <div
        className="card"
        style={{
          padding: 'var(--space-sm) var(--space-md)',
          background: 'var(--color-paper-2)',
          border: 'var(--rule-hair) solid var(--color-rule)',
        }}
      >
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <TrendingUp size={13} style={{ color: isRising ? 'var(--color-accent)' : 'var(--color-cool)' }} />
          Net Degradation Drift
        </div>
        <div
          style={{
            fontSize: 'var(--text-lg)',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            color: isRising ? 'var(--color-accent)' : 'var(--color-cool)',
            marginTop: 2,
          }}
        >
          {isRising ? `+${stats.netChange.toFixed(3)}` : stats.netChange.toFixed(3)}{' '}
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 400 }}>{sensorSpec.unit}</span>
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 3 }}>
          {stats.startValue.toFixed(1)} <ArrowRight size={10} /> {stats.endValue.toFixed(1)}
        </div>
      </div>
    </div>
  );
};
