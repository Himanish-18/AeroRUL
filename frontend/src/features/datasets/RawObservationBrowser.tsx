import React, { useState, useMemo } from 'react';
import {
  DatasetSplit,
  RawObservationRecord,
  SensorSpecification,
} from '../../lib/data/datasetExplorerTypes';
import { filterObservations } from '../../lib/data/datasetExplorerAdapter';
import { DatasetId } from '../../lib/data/types';
import {
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
} from 'lucide-react';

interface RawObservationBrowserProps {
  dataset: DatasetId;
  split: DatasetSplit;
  onSelectSplit: (split: DatasetSplit) => void;
  availableUnits: number[];
  selectedUnitId: number;
  onSelectUnit: (unitId: number) => void;
  rawObservations: RawObservationRecord[];
  isLoading: boolean;
  sensorSpecs: Record<string, SensorSpecification>;
}

const DEFAULT_SENSORS = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12'];

export const RawObservationBrowser: React.FC<RawObservationBrowserProps> = ({
  dataset,
  split,
  onSelectSplit,
  availableUnits,
  selectedUnitId,
  onSelectUnit,
  rawObservations,
  isLoading,
  sensorSpecs,
}) => {
  const [minCycleInput, setMinCycleInput] = useState<string>('');
  const [maxCycleInput, setMaxCycleInput] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('cycle');
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(25);
  const [selectedSensors, setSelectedSensors] = useState<string[]>(DEFAULT_SENSORS);
  const [showColumnModal, setShowColumnModal] = useState<boolean>(false);

  // Parse cycle bounds
  const minCycle = minCycleInput.trim() !== '' ? Number(minCycleInput) : undefined;
  const maxCycle = maxCycleInput.trim() !== '' ? Number(maxCycleInput) : undefined;

  // Filter observations
  const {
    items,
    totalCount,
    totalPages,
    allEngineCycles,
    minObservedCycle,
    maxObservedCycle,
  } = useMemo(() => {
    return filterObservations(rawObservations, {
      unitId: selectedUnitId,
      minCycle,
      maxCycle,
      selectedSensors,
      sortBy,
      sortAsc,
      page,
      pageSize,
    });
  }, [rawObservations, selectedUnitId, minCycle, maxCycle, selectedSensors, sortBy, sortAsc, page, pageSize]);

  // Toggle sensor column
  const toggleSensor = (sensorKey: string) => {
    setSelectedSensors((prev) =>
      prev.includes(sensorKey) ? prev.filter((s) => s !== sensorKey) : [...prev, sensorKey]
    );
  };

  const handleHeaderSort = (colKey: string) => {
    if (sortBy === colKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(colKey);
      setSortAsc(true);
    }
  };

  const resetFilters = () => {
    setMinCycleInput('');
    setMaxCycleInput('');
    setSortBy('cycle');
    setSortAsc(true);
    setPage(1);
    setSelectedSensors(DEFAULT_SENSORS);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Control Bar */}
      <div
        className="panel"
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-sm)',
          background: 'var(--color-paper-2)',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 'var(--space-md)',
          }}
        >
          {/* Split & Unit Selection */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
            <div className="segmented-control" role="tablist">
              <button
                type="button"
                className={`segmented-button ${split === 'train' ? 'active' : ''}`}
                onClick={() => {
                  onSelectSplit('train');
                  setPage(1);
                }}
                role="tab"
                aria-selected={split === 'train'}
              >
                Training
              </button>
              <button
                type="button"
                className={`segmented-button ${split === 'test' ? 'active' : ''}`}
                onClick={() => {
                  onSelectSplit('test');
                  setPage(1);
                }}
                role="tab"
                aria-selected={split === 'test'}
              >
                Test
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Engine:</span>
              <select
                value={selectedUnitId}
                onChange={(e) => {
                  onSelectUnit(Number(e.target.value));
                  setPage(1);
                }}
                className="select-field"
                style={{ fontSize: 'var(--text-xs)', height: '30px', fontWeight: 600 }}
                aria-label="Select Engine Unit"
                id="select-raw-unit"
              >
                {availableUnits.map((u) => (
                  <option key={u} value={u}>
                    Unit #{u}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Cycle Range and Column Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: 'var(--text-xs)' }}>
              <span style={{ color: 'var(--color-muted)' }}>Cycles:</span>
              <input
                type="number"
                placeholder={minObservedCycle ? String(minObservedCycle) : '1'}
                value={minCycleInput}
                onChange={(e) => {
                  setMinCycleInput(e.target.value);
                  setPage(1);
                }}
                className="input-field"
                style={{ width: '60px', height: '28px', fontSize: 'var(--text-xs)' }}
                aria-label="Minimum cycle filter"
              />
              <span style={{ color: 'var(--color-muted)' }}>to</span>
              <input
                type="number"
                placeholder={maxObservedCycle ? String(maxObservedCycle) : 'max'}
                value={maxCycleInput}
                onChange={(e) => {
                  setMaxCycleInput(e.target.value);
                  setPage(1);
                }}
                className="input-field"
                style={{ width: '60px', height: '28px', fontSize: 'var(--text-xs)' }}
                aria-label="Maximum cycle filter"
              />
            </div>

            {/* Column Visibility Trigger */}
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setShowColumnModal(!showColumnModal)}
              style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: 'var(--text-xs)', height: '28px' }}
              aria-expanded={showColumnModal}
              id="btn-column-selector"
            >
              <SlidersHorizontal size={13} />
              Sensors ({selectedSensors.length}/21)
            </button>

            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={resetFilters}
              title="Reset observation filters"
              style={{ fontSize: 'var(--text-xs)', height: '28px', padding: '0 6px' }}
              aria-label="Reset observation filters"
            >
              <RotateCcw size={13} />
            </button>
          </div>
        </div>

        {/* Column Selection Dropdown Drawer */}
        {showColumnModal && (
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule-2)',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-xs)',
              fontSize: 'var(--text-xs)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, color: 'var(--color-ink)' }}>Select Visible Sensor Columns:</span>
              <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                <button
                  type="button"
                  className="btn btn-ghost btn-xs"
                  onClick={() => setSelectedSensors(Array.from({ length: 21 }, (_, i) => `sensor_${i + 1}`))}
                >
                  Select All
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-xs"
                  onClick={() => setSelectedSensors(DEFAULT_SENSORS)}
                >
                  Reset Defaults
                </button>
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
                gap: '4px',
                maxHeight: '160px',
                overflowY: 'auto',
                paddingRight: '4px',
              }}
            >
              {Array.from({ length: 21 }, (_, i) => {
                const sKey = `sensor_${i + 1}`;
                const isChecked = selectedSensors.includes(sKey);
                const spec = sensorSpecs[sKey];
                return (
                  <label
                    key={sKey}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer',
                      padding: '3px 6px',
                      borderRadius: 'var(--radius-sm)',
                      background: isChecked ? 'var(--color-paper-2)' : 'transparent',
                      color: isChecked ? 'var(--color-ink)' : 'var(--color-muted)',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggleSensor(sKey)}
                      style={{ cursor: 'pointer' }}
                    />
                    <span style={{ fontFamily: 'var(--font-outlier)' }}>{sKey}</span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--color-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      ({spec ? spec.name.slice(0, 15) : ''})
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Observation Table Container */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        {/* Table Header metadata */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
            <span
              className="chip chip-ok"
              style={{ fontSize: '0.7rem', fontWeight: 600 }}
            >
              Raw C-MAPSS observation
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Engine Unit #{selectedUnitId} · {totalCount} of {allEngineCycles} recorded cycles matching
            </span>
          </div>

          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Showing rows {items.length > 0 ? (page - 1) * pageSize + 1 : 0} to{' '}
            {Math.min(page * pageSize, totalCount)} of {totalCount}
          </div>
        </div>

        {/* Contained Scroll Table */}
        <div
          style={{
            overflowX: 'auto',
            maxWidth: '100%',
            border: 'var(--rule-hair) solid var(--color-rule)',
            borderRadius: 'var(--radius-sm)',
          }}
        >
          {isLoading ? (
            <div style={{ padding: '48px', textAlign: 'center', color: 'var(--color-muted)', fontSize: 'var(--text-sm)' }}>
              Loading and decompressing raw observations for {dataset} ({split})…
            </div>
          ) : (
            <table
              className="table"
              style={{
                width: '100%',
                fontSize: 'var(--text-xs)',
                borderCollapse: 'collapse',
                whiteSpace: 'nowrap',
              }}
            >
              <thead>
                <tr style={{ background: 'var(--color-paper-3)', borderBottom: 'var(--rule-hair) solid var(--color-rule-2)' }}>
                  <th
                    onClick={() => handleHeaderSort('unit_id')}
                    style={{ padding: '8px 10px', textAlign: 'left', cursor: 'pointer', color: 'var(--color-muted)' }}
                  >
                    unit_id {sortBy === 'unit_id' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th
                    onClick={() => handleHeaderSort('cycle')}
                    style={{ padding: '8px 10px', textAlign: 'right', cursor: 'pointer', color: 'var(--color-accent)' }}
                  >
                    cycle {sortBy === 'cycle' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th
                    onClick={() => handleHeaderSort('op_setting_1')}
                    style={{ padding: '8px 10px', textAlign: 'right', cursor: 'pointer', color: 'var(--color-muted)' }}
                  >
                    op_setting_1 {sortBy === 'op_setting_1' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th
                    onClick={() => handleHeaderSort('op_setting_2')}
                    style={{ padding: '8px 10px', textAlign: 'right', cursor: 'pointer', color: 'var(--color-muted)' }}
                  >
                    op_setting_2 {sortBy === 'op_setting_2' && (sortAsc ? '▲' : '▼')}
                  </th>
                  <th
                    onClick={() => handleHeaderSort('op_setting_3')}
                    style={{ padding: '8px 10px', textAlign: 'right', cursor: 'pointer', color: 'var(--color-muted)' }}
                  >
                    op_setting_3 {sortBy === 'op_setting_3' && (sortAsc ? '▲' : '▼')}
                  </th>
                  {selectedSensors.map((sKey) => {
                    const spec = sensorSpecs[sKey];
                    return (
                      <th
                        key={sKey}
                        onClick={() => handleHeaderSort(sKey)}
                        style={{ padding: '8px 10px', textAlign: 'right', cursor: 'pointer', color: 'var(--color-muted)' }}
                        title={spec ? `${spec.name} (${spec.unit})` : sKey}
                      >
                        {sKey} {sortBy === sKey && (sortAsc ? '▲' : '▼')}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {items.length === 0 ? (
                  <tr>
                    <td
                      colSpan={5 + selectedSensors.length}
                      style={{ padding: '32px', textAlign: 'center', color: 'var(--color-muted)' }}
                    >
                      No raw observation rows found for Unit #{selectedUnitId} matching current cycle filters.
                    </td>
                  </tr>
                ) : (
                  items.map((row) => (
                    <tr
                      key={`${row.unit_id}_${row.cycle}`}
                      style={{
                        borderBottom: 'var(--rule-hair) solid var(--color-rule)',
                      }}
                    >
                      <td style={{ padding: '6px 10px', color: 'var(--color-ink)' }}>{row.unit_id}</td>
                      <td
                        className="measured"
                        style={{
                          padding: '6px 10px',
                          textAlign: 'right',
                          color: 'var(--color-accent)',
                          fontWeight: 600,
                        }}
                      >
                        {row.cycle}
                      </td>
                      <td className="measured" style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--color-ink-2)' }}>
                        {row.op_setting_1.toFixed(4)}
                      </td>
                      <td className="measured" style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--color-ink-2)' }}>
                        {row.op_setting_2.toFixed(4)}
                      </td>
                      <td className="measured" style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--color-ink-2)' }}>
                        {row.op_setting_3.toFixed(1)}
                      </td>
                      {selectedSensors.map((sKey) => {
                        const val = row[sKey];
                        return (
                          <td
                            key={sKey}
                            className="measured"
                            style={{ padding: '6px 10px', textAlign: 'right', color: 'var(--color-ink)' }}
                          >
                            {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) : '—'}
                          </td>
                        );
                      })}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination Bar */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            paddingTop: 'var(--space-xs)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Page Size:</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
              className="select-field"
              style={{ fontSize: 'var(--text-xs)', height: '26px' }}
              aria-label="Rows per page"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <button
              type="button"
              className="btn btn-secondary btn-xs"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="measured" style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Page {page} of {totalPages}
            </span>
            <button
              type="button"
              className="btn btn-secondary btn-xs"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              aria-label="Next page"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
