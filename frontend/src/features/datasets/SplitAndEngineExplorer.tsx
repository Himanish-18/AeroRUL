import React, { useState, useMemo } from 'react';
import {
  DatasetSplit,
  EngineFilterOptions,
  EngineIndexRecord,
} from '../../lib/data/datasetExplorerTypes';
import { filterAndSortEngines } from '../../lib/data/datasetExplorerAdapter';
import { DatasetId } from '../../lib/data/types';
import { Search, ArrowUpDown, ChevronLeft, ChevronRight, Eye } from 'lucide-react';

interface SplitAndEngineExplorerProps {
  dataset: DatasetId;
  split: DatasetSplit;
  onSelectSplit: (split: DatasetSplit) => void;
  engineIndex: EngineIndexRecord[];
  selectedUnitId: number;
  onSelectEngine: (unitId: number) => void;
  onInspectEngineInRaw: (unitId: number) => void;
}

export const SplitAndEngineExplorer: React.FC<SplitAndEngineExplorerProps> = ({
  dataset,
  split,
  onSelectSplit,
  engineIndex,
  selectedUnitId,
  onSelectEngine,
  onInspectEngineInRaw,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<EngineFilterOptions['sortBy']>('unit_id');
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  // Filter & paginate
  const { items, totalCount, totalPages } = useMemo(() => {
    return filterAndSortEngines(engineIndex, {
      searchQuery,
      sortBy,
      sortAsc,
      page,
      pageSize,
    });
  }, [engineIndex, searchQuery, sortBy, sortAsc, page, pageSize]);

  // Selected engine record
  const selectedEngine = useMemo(() => {
    return engineIndex.find((e) => e.unit_id === selectedUnitId) || engineIndex[0] || null;
  }, [engineIndex, selectedUnitId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Controls Bar: Split Selector & Search */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 'var(--space-md)',
        }}
      >
        {/* Split segmented control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', marginRight: '4px' }}>
            Fleet Split:
          </span>
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
              id="split-btn-train"
            >
              Training ({engineIndex.length > 0 && split === 'train' ? engineIndex.length : 'Run-to-failure'})
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
              id="split-btn-test"
            >
              Test ({engineIndex.length > 0 && split === 'test' ? engineIndex.length : 'Censored'})
            </button>
          </div>
        </div>

        {/* Search & Sort Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', flexWrap: 'wrap' }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search
              size={14}
              style={{ position: 'absolute', left: '10px', color: 'var(--color-muted)' }}
            />
            <input
              type="text"
              placeholder="Search engine #…"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              className="input-field"
              style={{
                paddingLeft: '28px',
                fontSize: 'var(--text-xs)',
                width: '150px',
                height: '32px',
              }}
              aria-label="Filter engines by Unit ID"
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as EngineFilterOptions['sortBy'])}
              className="select-field"
              style={{ fontSize: 'var(--text-xs)', height: '32px' }}
              aria-label="Sort engines by field"
            >
              <option value="unit_id">Sort: Unit ID</option>
              <option value="observation_count">Sort: Trajectory Length</option>
              <option value="last_observed_cycle">Sort: Last Cycle</option>
              {split === 'test' && <option value="official_endpoint_rul">Sort: Official RUL</option>}
            </select>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setSortAsc(!sortAsc)}
              title={sortAsc ? 'Ascending' : 'Descending'}
              style={{ height: '32px', padding: '0 8px' }}
              aria-label="Toggle sort order"
            >
              <ArrowUpDown size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Grid: Engine Table + Inspector Panel */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.8fr) minmax(280px, 1.2fr)',
          gap: 'var(--space-md)',
          alignItems: 'start',
        }}
      >
        {/* Engine Index Table */}
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
              {split === 'train' ? 'Training Fleet Engines' : 'Test Fleet Engines'} ({totalCount} matching)
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Page {page} of {totalPages}
            </span>
          </div>

          <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
            <table className="table" style={{ width: '100%', fontSize: 'var(--text-xs)', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: 'var(--rule-hair) solid var(--color-rule-2)', textAlign: 'left' }}>
                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)' }}
                    aria-sort={sortBy === 'unit_id' ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (sortBy === 'unit_id') setSortAsc(!sortAsc);
                        else { setSortBy('unit_id'); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: sortBy === 'unit_id' ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by Unit ID, ${sortBy === 'unit_id' ? (sortAsc ? 'ascending' : 'descending') : 'none'}`}
                    >
                      Unit ID {sortBy === 'unit_id' && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>

                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)', textAlign: 'right' }}
                    aria-sort={sortBy === 'observation_count' ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (sortBy === 'observation_count') setSortAsc(!sortAsc);
                        else { setSortBy('observation_count'); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: sortBy === 'observation_count' ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by Observations, ${sortBy === 'observation_count' ? (sortAsc ? 'ascending' : 'descending') : 'none'}`}
                    >
                      Observations {sortBy === 'observation_count' && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>

                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)', textAlign: 'right' }}
                    aria-sort={sortBy === 'first_cycle' ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (sortBy === 'first_cycle') setSortAsc(!sortAsc);
                        else { setSortBy('first_cycle'); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: sortBy === 'first_cycle' ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by First Cycle, ${sortBy === 'first_cycle' ? (sortAsc ? 'ascending' : 'descending') : 'none'}`}
                    >
                      First Cycle {sortBy === 'first_cycle' && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>

                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)', textAlign: 'right' }}
                    aria-sort={sortBy === 'last_observed_cycle' ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (sortBy === 'last_observed_cycle') setSortAsc(!sortAsc);
                        else { setSortBy('last_observed_cycle'); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: sortBy === 'last_observed_cycle' ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by Terminal Cycle, ${sortBy === 'last_observed_cycle' ? (sortAsc ? 'ascending' : 'descending') : 'none'}`}
                    >
                      Terminal Cycle {sortBy === 'last_observed_cycle' && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>

                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)' }}
                    aria-sort={sortBy === 'endpoint_status' ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        if (sortBy === 'endpoint_status') setSortAsc(!sortAsc);
                        else { setSortBy('endpoint_status'); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: sortBy === 'endpoint_status' ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by Status, ${sortBy === 'endpoint_status' ? (sortAsc ? 'ascending' : 'descending') : 'none'}`}
                    >
                      Status {sortBy === 'endpoint_status' && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>

                  <th
                    style={{ padding: '6px 8px', color: 'var(--color-muted)', textAlign: 'right' }}
                    aria-sort={(sortBy === 'physical_failure_cycle' || sortBy === 'official_endpoint_rul') ? (sortAsc ? 'ascending' : 'descending') : 'none'}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        const targetKey = split === 'train' ? 'physical_failure_cycle' : 'official_endpoint_rul';
                        if (sortBy === targetKey) setSortAsc(!sortAsc);
                        else { setSortBy(targetKey); setSortAsc(true); }
                        setPage(1);
                      }}
                      className="table-sort-header"
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        font: 'inherit',
                        fontSize: 'inherit',
                        color: (sortBy === 'physical_failure_cycle' || sortBy === 'official_endpoint_rul') ? 'var(--color-accent)' : 'var(--color-muted)',
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      aria-label={`Sort by ${split === 'train' ? 'Failure Cycle' : 'Official Endpoint RUL'}`}
                    >
                      {split === 'train' ? 'Failure Cycle' : 'Official Endpoint RUL'} {(sortBy === 'physical_failure_cycle' || sortBy === 'official_endpoint_rul') && (sortAsc ? '▲' : '▼')}
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ padding: '24px', textAlign: 'center', color: 'var(--color-muted)' }}>
                      No engines match search query "{searchQuery}"
                    </td>
                  </tr>
                ) : (
                  items.map((eng) => {
                    const isSelected = eng.unit_id === selectedUnitId;
                    return (
                      <tr
                        key={eng.unit_id}
                        onClick={() => onSelectEngine(eng.unit_id)}
                        style={{
                          cursor: 'pointer',
                          background: isSelected ? 'var(--color-paper-3)' : 'transparent',
                          borderBottom: 'var(--rule-hair) solid var(--color-rule)',
                          fontWeight: isSelected ? 600 : 400,
                        }}
                      >
                        <td style={{ padding: '6px 8px' }}>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectEngine(eng.unit_id);
                            }}
                            className="btn-select-engine"
                            aria-pressed={isSelected}
                            aria-label={`Select Engine Unit #${eng.unit_id}`}
                            style={{
                              background: 'none',
                              border: 'none',
                              padding: '2px 4px',
                              font: 'inherit',
                              fontSize: 'inherit',
                              color: isSelected ? 'var(--color-accent)' : 'var(--color-ink)',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              cursor: 'pointer',
                              borderRadius: 'var(--radius-xs)',
                              textAlign: 'left',
                            }}
                          >
                            {isSelected && (
                              <span
                                style={{
                                  width: '5px',
                                  height: '5px',
                                  borderRadius: '50%',
                                  background: 'var(--color-accent)',
                                }}
                              />
                            )}
                            Unit #{eng.unit_id}
                          </button>
                        </td>
                        <td className="measured" style={{ padding: '6px 8px', textAlign: 'right', color: 'var(--color-ink)' }}>
                          {eng.observation_count}
                        </td>
                        <td className="measured" style={{ padding: '6px 8px', textAlign: 'right', color: 'var(--color-muted)' }}>
                          {eng.first_cycle}
                        </td>
                        <td className="measured" style={{ padding: '6px 8px', textAlign: 'right', color: 'var(--color-ink-2)' }}>
                          {eng.last_observed_cycle}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <span
                            className={`chip ${eng.endpoint_status === 'run-to-failure' ? 'chip-warn' : 'chip-ok'}`}
                            style={{ fontSize: '0.65rem' }}
                          >
                            {eng.endpoint_status === 'run-to-failure' ? 'Run-to-failure' : 'Censored'}
                          </span>
                        </td>
                        <td className="measured" style={{ padding: '6px 8px', textAlign: 'right' }}>
                          {split === 'train' ? (
                            <span style={{ color: 'var(--color-ink-2)' }}>Cycle {eng.physical_failure_cycle}</span>
                          ) : (
                            <span style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                              {eng.official_endpoint_rul} cycles
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination bar */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: 'var(--space-xs)',
              borderTop: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Rows:</span>
              <select
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
                className="select-field"
                style={{ fontSize: 'var(--text-xs)', height: '26px', padding: '0 4px' }}
                aria-label="Engines per page"
              >
                <option value={10}>10</option>
                <option value={25}>25</option>
                <option value={50}>50</option>
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
                {page} / {totalPages}
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

        {/* Selected Engine Record Inspector */}
        {selectedEngine && (
          <div
            className="panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-md)',
              background: 'var(--color-paper-2)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                  Selected Engine Inspector
                </span>
                <h4 style={{ margin: 0, fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
                  Engine Unit #{selectedEngine.unit_id}
                </h4>
              </div>
              <span
                className={`chip ${selectedEngine.endpoint_status === 'run-to-failure' ? 'chip-warn' : 'chip-ok'}`}
              >
                {selectedEngine.endpoint_status === 'run-to-failure' ? 'Run-to-failure' : 'Censored'}
              </span>
            </div>

            {/* Spec grid */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 'var(--space-xs) var(--space-md)',
                fontSize: 'var(--text-xs)',
              }}
            >
              <span style={{ color: 'var(--color-muted)' }}>Dataset:</span>
              <span className="measured" style={{ color: 'var(--color-ink)' }}>{dataset}</span>

              <span style={{ color: 'var(--color-muted)' }}>Split:</span>
              <span style={{ color: 'var(--color-ink)', textTransform: 'capitalize' }}>{split}</span>

              <span style={{ color: 'var(--color-muted)' }}>Recorded observations:</span>
              <span className="measured" style={{ color: 'var(--color-ink)' }}>
                {selectedEngine.observation_count} cycles
              </span>

              <span style={{ color: 'var(--color-muted)' }}>First observed cycle:</span>
              <span className="measured" style={{ color: 'var(--color-ink)' }}>Cycle {selectedEngine.first_cycle}</span>

              <span style={{ color: 'var(--color-muted)' }}>Last observed cycle:</span>
              <span className="measured" style={{ color: 'var(--color-ink)' }}>Cycle {selectedEngine.last_observed_cycle}</span>

              {split === 'train' ? (
                <>
                  <span style={{ color: 'var(--color-muted)' }}>Failure endpoint cycle:</span>
                  <span className="measured" style={{ color: 'var(--color-warn)' }}>
                    Cycle {selectedEngine.physical_failure_cycle} (RUL = 0)
                  </span>

                  <span style={{ color: 'var(--color-muted)' }}>Derived initial RUL:</span>
                  <span className="measured" style={{ color: 'var(--color-accent)' }}>
                    {selectedEngine.last_observed_cycle - selectedEngine.first_cycle} cycles
                  </span>
                </>
              ) : (
                <>
                  <span style={{ color: 'var(--color-muted)' }}>Official endpoint RUL:</span>
                  <span className="measured" style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                    {selectedEngine.official_endpoint_rul} cycles
                  </span>

                  <span style={{ color: 'var(--color-muted)' }}>Censoring status:</span>
                  <span style={{ color: 'var(--color-ink-2)' }}>
                    Active prior to failure
                  </span>
                </>
              )}
            </div>

            {/* Semantic note */}
            <div
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-muted)',
                background: 'var(--color-paper-3)',
                padding: 'var(--space-sm)',
                borderRadius: 'var(--radius-sm)',
                lineHeight: 1.4,
              }}
            >
              {split === 'train' ? (
                <span>
                  <strong>Training RUL derivation:</strong> Physical RUL at any cycle is exactly{' '}
                  <code>{selectedEngine.last_observed_cycle} - cycle</code>. No arbitrary model clipping cap is applied here.
                </span>
              ) : (
                <span>
                  <strong>Endpoint-only ground truth:</strong> The official test RUL ({selectedEngine.official_endpoint_rul} cycles) applies strictly to cycle {selectedEngine.last_observed_cycle}. Test intermediate observations do not have ground truth.
                </span>
              )}
            </div>

            {/* Action button to switch to raw observation browser */}
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onInspectEngineInRaw(selectedEngine.unit_id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 'var(--space-xs)',
              }}
              id="btn-inspect-in-raw"
            >
              <Eye size={14} />
              Inspect raw cycles
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
