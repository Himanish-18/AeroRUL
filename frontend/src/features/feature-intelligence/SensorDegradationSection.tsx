import React, { useState, useMemo } from 'react';
import {
  DatasetId,
  SensorTrajectoryRecord,
  SensorTrajectoriesManifest,
} from '../../lib/data/types';
import {
  STATIC_SENSOR_SPECS,
  getSensorSpec,
} from '../../lib/data/featureIntelligenceAdapter';
import { SensorTrajectoryChart } from './SensorTrajectoryChart';
import { EngineSensorSummaryCards } from './EngineSensorSummaryCards';
import { DegradationInterpretationPanel } from './DegradationInterpretationPanel';
import { Gauge, RotateCcw, Users } from 'lucide-react';

interface SensorDegradationSectionProps {
  dataset: DatasetId;
  trajectories: SensorTrajectoryRecord[] | null;
  isLoadingTrajectories: boolean;
  manifest?: SensorTrajectoriesManifest | null;
}

export const SensorDegradationSection: React.FC<SensorDegradationSectionProps> = ({
  dataset,
  trajectories,
  isLoadingTrajectories,
  manifest,
}) => {
  // Available units for this dataset from manifest
  const subsetInfo = manifest?.subsets?.[dataset];
  const availableUnits = useMemo(() => {
    return subsetInfo?.selected_units || [1, 2, 3, 4, 5];
  }, [subsetInfo]);

  const defaultUnit = subsetInfo?.default_unit_id || availableUnits[0] || 1;

  // Active selections
  const [selectedSensor, setSelectedSensor] = useState<string>('sensor_11');
  const [selectedUnits, setSelectedUnits] = useState<number[]>([defaultUnit]);
  const [primaryUnitId, setPrimaryUnitId] = useState<number>(defaultUnit);

  // When dataset changes, sync default unit
  React.useEffect(() => {
    setSelectedUnits([defaultUnit]);
    setPrimaryUnitId(defaultUnit);
  }, [defaultUnit, dataset]);

  const toggleUnit = (unitId: number) => {
    if (selectedUnits.includes(unitId)) {
      if (selectedUnits.length > 1) {
        const next = selectedUnits.filter((u) => u !== unitId);
        setSelectedUnits(next);
        if (primaryUnitId === unitId) {
          setPrimaryUnitId(next[0]);
        }
      }
    } else {
      setSelectedUnits([...selectedUnits, unitId]);
      setPrimaryUnitId(unitId);
    }
  };

  const resetToDefault = () => {
    setSelectedUnits([defaultUnit]);
    setPrimaryUnitId(defaultUnit);
  };

  const compareAllEngines = () => {
    setSelectedUnits(availableUnits.slice(0, 5));
  };

  const sensorSpec = getSensorSpec(selectedSensor, manifest);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Controls Bar */}
      <div
        className="card"
        style={{
          padding: 'var(--space-md) var(--space-lg)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-md)',
          background: 'var(--color-paper-2)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
          {/* Sensor Dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', color: 'var(--color-muted)', fontSize: 'var(--text-xs)' }}>
              <Gauge size={14} style={{ color: 'var(--color-accent)' }} />
              Sensor:
            </div>
            <select
              value={selectedSensor}
              onChange={(e) => setSelectedSensor(e.target.value)}
              className="select"
              style={{
                fontSize: 'var(--text-xs)',
                background: 'var(--color-paper-3)',
                color: 'var(--color-ink)',
                border: 'var(--rule-hair) solid var(--color-rule)',
                borderRadius: 'var(--radius-sm)',
                padding: '6px 12px',
                minWidth: 280,
                fontFamily: 'var(--font-mono)',
              }}
              aria-label="Select sensor measurement"
            >
              {Object.keys(STATIC_SENSOR_SPECS).map((sKey) => {
                const s = STATIC_SENSOR_SPECS[sKey];
                return (
                  <option key={sKey} value={sKey}>
                    {sKey} — {s.name} ({s.unit})
                  </option>
                );
              })}
            </select>
          </div>

          {/* Quick Actions */}
          <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={resetToDefault}
              style={{ fontSize: 'var(--text-xs)', display: 'flex', alignItems: 'center', gap: 4 }}
              title="Reset to single default engine"
            >
              <RotateCcw size={12} />
              Reset (Unit {defaultUnit})
            </button>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={compareAllEngines}
              style={{ fontSize: 'var(--text-xs)', display: 'flex', alignItems: 'center', gap: 4 }}
              title="Overlay multiple engines (Units 1 to 5)"
            >
              <Users size={12} />
              Compare Fleet 1 to 5
            </button>
          </div>
        </div>

        {/* Engine Selection Chips */}
        <div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', marginBottom: 'var(--space-xs)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>Available Engines (click to overlay or compare):</span>
            <span style={{ fontSize: '0.65rem', color: 'var(--color-accent)' }}>
              {selectedUnits.length} active
            </span>
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-xs)', flexWrap: 'wrap' }}>
            {availableUnits.map((u) => {
              const isSelected = selectedUnits.includes(u);
              const meta = subsetInfo?.unit_metadata?.[String(u)];
              const lifetime = meta?.lifetime_cycles;
              return (
                <button
                  key={u}
                  type="button"
                  onClick={() => toggleUnit(u)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '4px 10px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: 'var(--text-xs)',
                    fontFamily: 'var(--font-mono)',
                    cursor: 'pointer',
                    background: isSelected ? 'var(--color-paper-3)' : 'var(--color-paper-1)',
                    color: isSelected ? 'var(--color-ink)' : 'var(--color-muted)',
                    border: isSelected
                      ? 'var(--rule-hair) solid var(--color-accent)'
                      : 'var(--rule-hair) solid var(--color-rule)',
                  }}
                  aria-pressed={isSelected}
                  aria-label={`Toggle Engine ${u}`}
                >
                  <span
                    style={{
                      width: 7,
                      height: 7,
                      borderRadius: '50%',
                      background: isSelected ? 'var(--color-accent)' : 'var(--color-muted)',
                    }}
                  />
                  <span>Engine {u}</span>
                  {lifetime && (
                    <span style={{ fontSize: '0.65rem', color: 'var(--color-muted)', fontWeight: 400 }}>
                      ({lifetime}c)
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Trajectory Workspace */}
      <div className="card" style={{ padding: 'var(--space-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-sm)' }}>
          <div>
            <h3 style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              {selectedSensor}: {sensorSpec.name} Trajectories
            </h3>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 'var(--space-xxs) 0 0 0' }}>
              Run-to-failure sensor progression across operating cycles for {selectedUnits.length} engine(s)
            </p>
          </div>
          <span
            style={{
              fontSize: 'var(--text-xs)',
              fontFamily: 'var(--font-mono)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--color-paper-3)',
              color: 'var(--color-ink)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            {sensorSpec.unit} · {sensorSpec.type}
          </span>
        </div>

        {/* Engine Summary Stats for Primary Selected Engine */}
        {trajectories && (
          <EngineSensorSummaryCards
            trajectories={trajectories}
            sensor={selectedSensor}
            primaryUnitId={primaryUnitId}
            manifest={manifest}
          />
        )}

        {/* Trajectory Line Chart */}
        {isLoadingTrajectories ? (
          <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-muted)' }}>
            Loading sensor trajectory series for {dataset}…
          </div>
        ) : trajectories ? (
          <SensorTrajectoryChart
            trajectories={trajectories}
            sensor={selectedSensor}
            selectedUnits={selectedUnits}
            manifest={manifest}
          />
        ) : (
          <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-muted)' }}>
            No trajectory data available.
          </div>
        )}
      </div>

      {/* Domain Interpretation & Boundaries */}
      <DegradationInterpretationPanel dataset={dataset} />
    </div>
  );
};
