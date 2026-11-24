import React, { useState, useMemo } from 'react';
import {
  DatasetSplit,
  RawObservationRecord,
  SensorStats,
  OpSettingStats,
  SensorSpecification,
  RulStats,
  EngineIndexRecord,
} from '../../lib/data/datasetExplorerTypes';
import {
  computeNumericDistribution,
  computeTestEndpointRulDistribution,
  DistributionBin,
} from '../../lib/data/datasetExplorerAdapter';
import { DatasetId } from '../../lib/data/types';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { Activity, Gauge, Sliders } from 'lucide-react';

interface SensorAndSettingsExplorerProps {
  dataset: DatasetId;
  split: DatasetSplit;
  onSelectSplit: (split: DatasetSplit) => void;
  rawObservations: RawObservationRecord[];
  sensorSummaries: Record<string, SensorStats>;
  opSettingSummaries: Record<string, OpSettingStats>;
  sensorSpecs: Record<string, SensorSpecification>;
  rulStats: RulStats;
  engineIndex?: EngineIndexRecord[];
  testEndpointRuls?: number[];
}

export const SensorAndSettingsExplorer: React.FC<SensorAndSettingsExplorerProps> = ({
  dataset,
  split,
  onSelectSplit,
  rawObservations,
  sensorSummaries,
  opSettingSummaries,
  sensorSpecs,
  rulStats,
  engineIndex = [],
  testEndpointRuls,
}) => {
  const [selectedSensor, setSelectedSensor] = useState<string>('sensor_4');
  const [activeSubTab, setActiveSubTab] = useState<'sensors' | 'settings' | 'rul'>('sensors');

  const currentSensorSummary = sensorSummaries[selectedSensor] || null;
  const currentSensorSpec = sensorSpecs[selectedSensor] || null;

  // Compute distribution for selected sensor from current loaded raw observations
  const sensorHistogramData = useMemo(() => {
    if (rawObservations.length === 0) return [];
    const values = rawObservations.map((r) => r[selectedSensor]).filter((v) => typeof v === 'number' && !isNaN(v));
    return computeNumericDistribution(values, 16);
  }, [rawObservations, selectedSensor]);

  // Compute 2D scatter sample of op_setting_1 (Altitude) vs op_setting_2 (Mach)
  const opScatterData = useMemo(() => {
    if (rawObservations.length === 0) return [];
    // Stride sample up to 400 points for crisp performant visualization
    const stride = Math.max(1, Math.floor(rawObservations.length / 400));
    const sample = [];
    for (let i = 0; i < rawObservations.length; i += stride) {
      sample.push({
        altitude: rawObservations[i].op_setting_1,
        mach: rawObservations[i].op_setting_2,
        tra: rawObservations[i].op_setting_3,
      });
    }
    return sample;
  }, [rawObservations]);

  // Compute RUL distribution histogram
  const rulHistogramData = useMemo(() => {
    if (split === 'train') {
      if (rawObservations.length === 0) return [];
      // Calculate derived physical RUL across all training observations
      const unitMaxCycles = new Map<number, number>();
      for (const r of rawObservations) {
        const cur = unitMaxCycles.get(r.unit_id) || 0;
        if (r.cycle > cur) unitMaxCycles.set(r.unit_id, r.cycle);
      }
      const derivedRuls: number[] = [];
      for (const r of rawObservations) {
        const maxC = unitMaxCycles.get(r.unit_id)!;
        derivedRuls.push(maxC - r.cycle);
      }
      return computeNumericDistribution(derivedRuls, 16);
    } else {
      // For test split: use actual official test endpoint RUL values
      // Exactly one endpoint RUL per test engine. Never propagate backwards across cycles.
      const testRecords = engineIndex.filter((e) => e.split === 'test' && e.official_endpoint_rul != null);
      if (testRecords.length > 0) {
        return computeTestEndpointRulDistribution(testRecords, 16);
      }
      if (testEndpointRuls && testEndpointRuls.length > 0) {
        return computeTestEndpointRulDistribution(testEndpointRuls, 16);
      }
      return [];
    }
  }, [rawObservations, split, engineIndex, testEndpointRuls]);

  const isMultiRegime = dataset === 'FD002' || dataset === 'FD004';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Subtab Navigator & Split Control */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 'var(--space-md)',
        }}
      >
        <div className="segmented-control" role="tablist">
          <button
            type="button"
            className={`segmented-button ${activeSubTab === 'sensors' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('sensors')}
            role="tab"
            aria-selected={activeSubTab === 'sensors'}
            id="subtab-sensors"
            style={{ whiteSpace: 'nowrap' }}
          >
            <Activity size={14} style={{ marginRight: '4px' }} />
            Sensors (21)
          </button>
          <button
            type="button"
            className={`segmented-button ${activeSubTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('settings')}
            role="tab"
            aria-selected={activeSubTab === 'settings'}
            id="subtab-settings"
            style={{ whiteSpace: 'nowrap' }}
          >
            <Sliders size={14} style={{ marginRight: '4px' }} />
            Settings (3)
          </button>
          <button
            type="button"
            className={`segmented-button ${activeSubTab === 'rul' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('rul')}
            role="tab"
            aria-selected={activeSubTab === 'rul'}
            id="subtab-rul"
            style={{ whiteSpace: 'nowrap' }}
          >
            <Gauge size={14} style={{ marginRight: '4px' }} />
            RUL Distribution
          </button>
        </div>

        {/* Split selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Split Context:</span>
          <div className="segmented-control">
            <button
              type="button"
              className={`segmented-button ${split === 'train' ? 'active' : ''}`}
              onClick={() => onSelectSplit('train')}
              id="btn-split-train"
            >
              Training Split
            </button>
            <button
              type="button"
              className={`segmented-button ${split === 'test' ? 'active' : ''}`}
              onClick={() => onSelectSplit('test')}
              id="btn-split-test"
            >
              Test Split
            </button>
          </div>
        </div>
      </div>

      {/* VIEW 1: SENSORS EXPLORER */}
      {activeSubTab === 'sensors' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {/* Sensor Dropdown Selector */}
          <div
            className="panel"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 'var(--space-md)',
              background: 'var(--color-paper-2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Select Sensor:</span>
              <select
                value={selectedSensor}
                onChange={(e) => setSelectedSensor(e.target.value)}
                className="select-field"
                style={{ fontSize: 'var(--text-xs)', height: '32px', minWidth: '280px', fontWeight: 600 }}
                aria-label="Select sensor to inspect"
                id="select-sensor-explorer"
              >
                {Array.from({ length: 21 }, (_, i) => {
                  const sKey = `sensor_${i + 1}`;
                  const spec = sensorSpecs[sKey];
                  const summary = sensorSummaries[sKey];
                  const nearConstTag = summary?.is_near_constant ? ' [Near-Constant]' : '';
                  return (
                    <option key={sKey} value={sKey}>
                      {sKey} — {spec ? spec.name : ''} ({spec ? spec.unit : ''}){nearConstTag}
                    </option>
                  );
                })}
              </select>
            </div>

            {currentSensorSummary && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                {currentSensorSummary.is_constant ? (
                  <span className="chip chip-bad" style={{ fontSize: 'var(--text-xs)' }}>
                    Constant Value
                  </span>
                ) : currentSensorSummary.is_near_constant ? (
                  <span className="chip chip-warn" style={{ fontSize: 'var(--text-xs)' }}>
                    Near-Constant (std/mean &lt; 0.001)
                  </span>
                ) : (
                  <span className="chip chip-ok" style={{ fontSize: 'var(--text-xs)' }}>
                    Active Measurement ({currentSensorSummary.n_unique} unique)
                  </span>
                )}
                <span className="chip" style={{ fontSize: 'var(--text-xs)' }}>
                  Type: {currentSensorSpec?.type}
                </span>
              </div>
            )}
          </div>

          {/* Sensor Profile & Histogram Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(260px, 1fr) minmax(0, 2fr)',
              gap: 'var(--space-md)',
              alignItems: 'start',
            }}
          >
            {/* Sensor Summary Metrics Card */}
            {currentSensorSummary && (
              <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                  {selectedSensor.toUpperCase()} Statistics ({dataset} {split})
                </span>
                <h3 style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
                  {currentSensorSpec?.name}
                </h3>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-accent)', fontWeight: 500 }}>
                  Unit: {currentSensorSpec?.unit}
                </span>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '4px 12px',
                    fontSize: 'var(--text-xs)',
                    marginTop: 'var(--space-xs)',
                  }}
                >
                  <span style={{ color: 'var(--color-muted)' }}>Minimum:</span>
                  <span className="measured" style={{ color: 'var(--color-ink)' }}>{currentSensorSummary.min}</span>

                  <span style={{ color: 'var(--color-muted)' }}>Maximum:</span>
                  <span className="measured" style={{ color: 'var(--color-ink)' }}>{currentSensorSummary.max}</span>

                  <span style={{ color: 'var(--color-muted)' }}>Mean:</span>
                  <span className="measured" style={{ color: 'var(--color-ink)' }}>{currentSensorSummary.mean}</span>

                  <span style={{ color: 'var(--color-muted)' }}>Std Deviation:</span>
                  <span className="measured" style={{ color: 'var(--color-ink)' }}>{currentSensorSummary.std}</span>

                  <span style={{ color: 'var(--color-muted)' }}>Unique Values:</span>
                  <span className="measured" style={{ color: 'var(--color-ink)' }}>{currentSensorSummary.n_unique}</span>

                  <span style={{ color: 'var(--color-muted)' }}>Missing Count:</span>
                  <span className="measured" style={{ color: 'var(--color-ok)' }}>{currentSensorSummary.missing_count} (0.0%)</span>
                </div>

                <div
                  style={{
                    marginTop: 'var(--space-xs)',
                    padding: 'var(--space-xs)',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--color-paper-2)',
                    fontSize: '0.7rem',
                    color: 'var(--color-muted)',
                    lineHeight: 1.4,
                  }}
                >
                  <strong>Measurement Notes:</strong> Statistics are computed strictly on the selected{' '}
                  <code>{split}</code> cohort ({rawObservations.length.toLocaleString()} rows). Training and test distributions are preserved without cross-split mixing or normalization leakage.
                </div>
              </div>
            )}

            {/* Sensor Histogram Chart */}
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
                    Observation Value Distribution ({selectedSensor})
                  </h4>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                    Equal-width histogram across all recorded cycles in {dataset} {split}
                  </span>
                </div>
                <span className="chip" style={{ fontSize: '0.65rem' }}>
                  {rawObservations.length.toLocaleString()} observations
                </span>
              </div>

              <div style={{ width: '100%', height: '280px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={sensorHistogramData}
                    margin={{ top: 12, right: 20, bottom: 24, left: 10 }}
                  >
                    <CartesianGrid strokeDasharray="0" stroke="var(--color-rule)" />
                    <XAxis
                      dataKey="binLabel"
                      tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                      stroke="var(--color-rule-2)"
                      label={{
                        value: `${currentSensorSpec?.name || selectedSensor} (${currentSensorSpec?.unit || ''})`,
                        position: 'insideBottom',
                        offset: -12,
                        fill: 'var(--color-muted)',
                        fontSize: 11,
                      }}
                    />
                    <YAxis
                      tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                      stroke="var(--color-rule-2)"
                      label={{
                        value: 'Count',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 10,
                        fill: 'var(--color-muted)',
                        fontSize: 11,
                        style: { textAnchor: 'middle' },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload || payload.length === 0) return null;
                        const data = payload[0].payload as DistributionBin;
                        return (
                          <div
                            style={{
                              background: 'var(--color-paper-3)',
                              border: 'var(--rule-hair) solid var(--color-rule-2)',
                              padding: '6px 10px',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: 'var(--text-xs)',
                            }}
                          >
                            <div style={{ color: 'var(--color-muted)' }}>
                              Range: <span className="measured" style={{ color: 'var(--color-ink)' }}>{data.binStart.toFixed(4)} to {data.binEnd.toFixed(4)}</span>
                            </div>
                            <div style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                              Count: {data.count.toLocaleString()} observations
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="count" fill="var(--color-accent)" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: OPERATING SETTINGS EXPLORER */}
      {activeSubTab === 'settings' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {/* Three Setting Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: 'var(--space-md)',
            }}
          >
            {['op_setting_1', 'op_setting_2', 'op_setting_3'].map((key) => {
              const s = opSettingSummaries[key];
              if (!s) return null;
              return (
                <div key={key} className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                    {key.toUpperCase()}
                  </span>
                  <h4 style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
                    {s.name}
                  </h4>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
                    {s.description}
                  </span>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '4px 8px',
                      fontSize: 'var(--text-xs)',
                      marginTop: 'var(--space-xs)',
                    }}
                  >
                    <span style={{ color: 'var(--color-muted)' }}>Range:</span>
                    <span className="measured" style={{ color: 'var(--color-ink)' }}>
                      {s.min} to {s.max}
                    </span>

                    <span style={{ color: 'var(--color-muted)' }}>Mean:</span>
                    <span className="measured" style={{ color: 'var(--color-ink)' }}>{s.mean}</span>

                    <span style={{ color: 'var(--color-muted)' }}>Std:</span>
                    <span className="measured" style={{ color: 'var(--color-ink)' }}>{s.std}</span>

                    <span style={{ color: 'var(--color-muted)' }}>Unique values:</span>
                    <span className="measured" style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                      {s.n_unique}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Setting 1 vs Setting 2 Scatter (Altitude vs Mach) */}
          <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
              <div>
                <h4 style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
                  Flight Regime Space: Altitude (Setting 1) vs Mach (Setting 2)
                </h4>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                  {isMultiRegime
                    ? `${dataset} contains six distinct flight operating condition clusters spanning altitude and Mach regimes.`
                    : `${dataset} is designed as a single operating condition near sea level with near-constant settings.`}
                </span>
              </div>
              <span className={`chip ${isMultiRegime ? 'chip-warn' : 'chip-ok'}`}>
                {isMultiRegime ? '6 Operating Conditions' : '1 Condition (Sea Level)'}
              </span>
            </div>

            <div style={{ width: '100%', height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 10 }}>
                  <CartesianGrid strokeDasharray="0" stroke="var(--color-rule)" />
                  <XAxis
                    type="number"
                    dataKey="altitude"
                    name="Altitude"
                    tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                    stroke="var(--color-rule-2)"
                    label={{
                      value: 'op_setting_1 (Altitude in kft)',
                      position: 'insideBottom',
                      offset: -12,
                      fill: 'var(--color-muted)',
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="mach"
                    name="Mach Number"
                    tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                    stroke="var(--color-rule-2)"
                    label={{
                      value: 'op_setting_2 (Mach)',
                      angle: -90,
                      position: 'insideLeft',
                      offset: 10,
                      fill: 'var(--color-muted)',
                      fontSize: 11,
                      style: { textAnchor: 'middle' },
                    }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || payload.length === 0) return null;
                      const pt = payload[0].payload;
                      return (
                        <div
                          style={{
                            background: 'var(--color-paper-3)',
                            border: 'var(--rule-hair) solid var(--color-rule-2)',
                            padding: '6px 10px',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: 'var(--text-xs)',
                          }}
                        >
                          <div>Setting 1 (Altitude): <span className="measured">{pt.altitude.toFixed(4)}</span></div>
                          <div>Setting 2 (Mach): <span className="measured">{pt.mach.toFixed(4)}</span></div>
                          <div>Setting 3 (TRA): <span className="measured">{pt.tra.toFixed(1)}</span></div>
                        </div>
                      );
                    }}
                  />
                  <Scatter name="Operating Points" data={opScatterData} fill="var(--color-cool)" opacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 3: RUL DISTRIBUTION */}
      {activeSubTab === 'rul' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(280px, 1fr) minmax(0, 2fr)',
              gap: 'var(--space-md)',
              alignItems: 'start',
            }}
          >
            {/* RUL Semantic Card */}
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                RUL Ground Truth Semantics
              </span>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
                {split === 'train' ? 'Training Physical RUL' : 'Official Test Endpoint RUL'}
              </h3>
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', margin: 0, lineHeight: 1.5 }}>
                {rulStats.description}
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '4px 8px',
                  fontSize: 'var(--text-xs)',
                  marginTop: 'var(--space-xs)',
                }}
              >
                <span style={{ color: 'var(--color-muted)' }}>Count:</span>
                <span className="measured" style={{ color: 'var(--color-ink)' }}>
                  {rulStats.observation_count || rulStats.endpoint_count} {split === 'train' ? 'cycles' : 'engines'}
                </span>

                <span style={{ color: 'var(--color-muted)' }}>Minimum:</span>
                <span className="measured" style={{ color: 'var(--color-ink)' }}>{rulStats.min} cycles</span>

                <span style={{ color: 'var(--color-muted)' }}>Maximum:</span>
                <span className="measured" style={{ color: 'var(--color-ink)' }}>{rulStats.max} cycles</span>

                <span style={{ color: 'var(--color-muted)' }}>Mean:</span>
                <span className="measured" style={{ color: 'var(--color-accent)' }}>{rulStats.mean} cycles</span>

                <span style={{ color: 'var(--color-muted)' }}>Median:</span>
                <span className="measured" style={{ color: 'var(--color-ink)' }}>{rulStats.median} cycles</span>

                <span style={{ color: 'var(--color-muted)' }}>Std Deviation:</span>
                <span className="measured" style={{ color: 'var(--color-ink)' }}>{rulStats.std} cycles</span>
              </div>
            </div>

            {/* RUL Distribution Chart */}
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
              <h4 style={{ margin: 0, fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
                {split === 'train'
                  ? `Derived Physical Training RUL Distribution (${dataset})`
                  : `Official Ground Truth Test Endpoint RUL (${dataset})`}
              </h4>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                {split === 'train'
                  ? 'Calculated per cycle as (failure_cycle - cycle). Reaches exactly 0 at failure.'
                  : `Single official endpoint label per engine for all ${rulStats.endpoint_count || 100} test engines. Official C-MAPSS test labels are specified only for the final censored observation cycle; RUL is not defined per-cycle during test operation.`}
              </span>

              <div style={{ width: '100%', height: '280px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rulHistogramData} margin={{ top: 12, right: 20, bottom: 24, left: 10 }}>
                    <CartesianGrid strokeDasharray="0" stroke="var(--color-rule)" />
                    <XAxis
                      dataKey="binLabel"
                      tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                      stroke="var(--color-rule-2)"
                      label={{
                        value: 'Remaining Useful Life (cycles)',
                        position: 'insideBottom',
                        offset: -12,
                        fill: 'var(--color-muted)',
                        fontSize: 11,
                      }}
                    />
                    <YAxis
                      tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-outlier)' }}
                      stroke="var(--color-rule-2)"
                      label={{
                        value: split === 'train' ? 'Observations' : 'Test Engines',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 10,
                        fill: 'var(--color-muted)',
                        fontSize: 11,
                        style: { textAnchor: 'middle' },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload || payload.length === 0) return null;
                        const data = payload[0].payload as DistributionBin;
                        return (
                          <div
                            style={{
                              background: 'var(--color-paper-3)',
                              border: 'var(--rule-hair) solid var(--color-rule-2)',
                              padding: '6px 10px',
                              borderRadius: 'var(--radius-sm)',
                              fontSize: 'var(--text-xs)',
                            }}
                          >
                            <div>RUL Range: <span className="measured">{data.binLabel} cycles</span></div>
                            <div style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                              Count: {data.count.toLocaleString()} {split === 'train' ? 'observations' : 'test engines'}
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Bar
                      dataKey="count"
                      fill={split === 'train' ? 'var(--color-cool)' : 'var(--color-ok)'}
                      radius={[2, 2, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
