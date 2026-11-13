import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { SensorTrajectoryRecord, SensorTrajectoriesManifest } from '../../lib/data/types';
import {
  getTrajectoryChartData,
  getSensorSpec,
  getEngineColor,
} from '../../lib/data/featureIntelligenceAdapter';

interface SensorTrajectoryChartProps {
  trajectories: SensorTrajectoryRecord[];
  sensor: string;
  selectedUnits: number[];
  manifest?: SensorTrajectoriesManifest | null;
}

export const SensorTrajectoryChart: React.FC<SensorTrajectoryChartProps> = ({
  trajectories,
  sensor,
  selectedUnits,
  manifest,
}) => {
  const chartData = getTrajectoryChartData(trajectories, sensor, selectedUnits);
  const sensorSpec = getSensorSpec(sensor, manifest);

  if (chartData.length === 0 || selectedUnits.length === 0) {
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
        Select at least one engine and sensor to visualize degradation dynamics.
      </div>
    );
  }

  // Calculate domain padding
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (const pt of chartData) {
    for (const u of selectedUnits) {
      const val = pt[`unit_${u}`];
      if (typeof val === 'number') {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
  }

  const range = maxVal - minVal;
  const padding = range === 0 ? Math.abs(minVal) * 0.05 || 1 : range * 0.05;
  const domainY: [number, number] = [
    Number((minVal - padding).toFixed(3)),
    Number((maxVal + padding).toFixed(3)),
  ];

  return (
    <div
      style={{ width: '100%', height: 380 }}
      data-testid="sensor-trajectory-chart"
      aria-label={`Sensor trajectory chart for ${sensor}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 15, right: 25, left: 10, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-rule)" />
          <XAxis
            dataKey="cycle"
            type="number"
            domain={['dataMin', 'dataMax']}
            tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--color-rule)' }}
            label={{
              value: 'Engine Operating Cycles (run-to-failure progression)',
              position: 'bottom',
              offset: 12,
              fill: 'var(--color-muted)',
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            domain={domainY}
            tick={{ fill: 'var(--color-muted)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--color-rule)' }}
            label={{
              value: `${sensorSpec.name} (${sensorSpec.unit})`,
              angle: -90,
              position: 'left',
              offset: -5,
              fill: 'var(--color-muted)',
              fontSize: 11,
            }}
            tickFormatter={(val: number) =>
              Math.abs(val) >= 1000 ? val.toLocaleString(undefined, { maximumFractionDigits: 0 }) : val.toFixed(2)
            }
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload || !payload.length) return null;
              return (
                <div
                  style={{
                    background: 'var(--color-paper-2)',
                    border: 'var(--rule-hair) solid var(--color-rule)',
                    padding: 'var(--space-xs) var(--space-sm)',
                    borderRadius: 'var(--radius-sm)',
                    boxShadow: 'var(--shadow-md)',
                    fontSize: 'var(--text-xs)',
                  }}
                >
                  <div
                    style={{
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--color-ink)',
                      fontWeight: 600,
                      marginBottom: 'var(--space-xxs)',
                    }}
                  >
                    Cycle #{label}
                  </div>
                  {payload.map((entry) => {
                    const unitId = String(entry.name).replace('unit_', '');
                    return (
                      <div
                        key={entry.name}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 'var(--space-xs)',
                          marginBottom: 2,
                        }}
                      >
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background: entry.color,
                            display: 'inline-block',
                          }}
                        />
                        <span style={{ color: 'var(--color-muted)' }}>Engine {unitId}:</span>
                        <strong
                          style={{
                            fontFamily: 'var(--font-mono)',
                            color: entry.color,
                          }}
                        >
                          {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}{' '}
                          <span style={{ fontSize: '0.65rem', fontWeight: 400 }}>{sensorSpec.unit}</span>
                        </strong>
                      </div>
                    );
                  })}
                </div>
              );
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{ paddingBottom: 10, fontSize: '0.75rem' }}
            formatter={(val: string) => {
              const u = val.replace('unit_', '');
              return <span style={{ color: 'var(--color-ink)', fontFamily: 'var(--font-mono)' }}>Unit {u}</span>;
            }}
          />
          {selectedUnits.map((unitId, idx) => (
            <Line
              key={unitId}
              type="monotone"
              dataKey={`unit_${unitId}`}
              stroke={getEngineColor(idx)}
              strokeWidth={1.7}
              dot={false}
              activeDot={{ r: 4, fill: getEngineColor(idx), stroke: 'var(--color-paper-1)' }}
              name={`unit_${unitId}`}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
