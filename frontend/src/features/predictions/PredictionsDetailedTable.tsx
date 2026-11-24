import React, { useState, useMemo, useEffect } from 'react';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Eye,
} from 'lucide-react';
import { PredictionRecord, SortField, SortDirection } from '../../lib/data/types';
import { formatCycles, formatSignedError } from '../../lib/utils/format';
import { EmptyState } from '../../components/shared/EmptyState';

interface PredictionsDetailedTableProps {
  predictions: PredictionRecord[];
  selectedUnitId: number | null;
  onSelectEngine: (unitId: number) => void;
  onResetFilters?: () => void;
}

export const PredictionsDetailedTable: React.FC<PredictionsDetailedTableProps> = ({
  predictions,
  selectedUnitId,
  onSelectEngine,
  onResetFilters,
}) => {
  const [sortField, setSortField] = useState<SortField>('unit_id');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Task 4: Reset current page to 1 whenever incoming predictions cohort changes
  useEffect(() => {
    setCurrentPage(1);
  }, [predictions]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
    setCurrentPage(1);
  };

  const sortedData = useMemo(() => {
    return [...predictions].sort((a, b) => {
      const valA = a[sortField];
      const valB = b[sortField];

      if (typeof valA === 'number' && typeof valB === 'number') {
        return sortDirection === 'asc' ? valA - valB : valB - valA;
      }

      const strA = String(valA);
      const strB = String(valB);
      return sortDirection === 'asc'
        ? strA.localeCompare(strB)
        : strB.localeCompare(strA);
    });
  }, [predictions, sortField, sortDirection]);

  const totalPages = Math.max(1, Math.ceil(sortedData.length / pageSize));
  // Clamp page number to avoid stale out-of-range pagination
  const safeCurrentPage = Math.min(Math.max(1, currentPage), totalPages);

  const paginatedData = useMemo(() => {
    const start = (safeCurrentPage - 1) * pageSize;
    return sortedData.slice(start, start + pageSize);
  }, [sortedData, safeCurrentPage, pageSize]);

  const getAriaSort = (field: SortField): 'ascending' | 'descending' | 'none' => {
    if (sortField !== field) return 'none';
    return sortDirection === 'asc' ? 'ascending' : 'descending';
  };

  const renderSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown size={12} color="var(--color-rule-2)" aria-hidden="true" />;
    }
    return sortDirection === 'asc' ? (
      <ArrowUp size={12} color="var(--color-accent)" aria-hidden="true" />
    ) : (
      <ArrowDown size={12} color="var(--color-accent)" aria-hidden="true" />
    );
  };

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Table Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Official Test Engine Predictions
          </h3>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Showing {sortedData.length} engine endpoints · Select an engine to inspect prediction error breakdown
          </span>
        </div>

        {/* Page Size Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
          <span>Rows per page:</span>
          <select
            className="input-text"
            style={{ height: '30px', paddingRight: '20px', fontSize: 'var(--text-xs)' }}
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            aria-label="Rows per page"
          >
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>

      {/* Table / Empty State */}
      {sortedData.length === 0 ? (
        <EmptyState
          title="No engines match current filter"
          description="Adjust or reset search query and error thresholds to view engine predictions."
          actionText={onResetFilters ? 'Reset all filters' : undefined}
          onAction={onResetFilters}
        />
      ) : (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th aria-sort={getAriaSort('unit_id')}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('unit_id')}
                    aria-label={`Sort by Engine ID, currently ${getAriaSort('unit_id')}`}
                  >
                    <span>Engine ID</span>
                    {renderSortIcon('unit_id')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('last_observed_cycle')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('last_observed_cycle')}
                    aria-label={`Sort by Last observed cycle, currently ${getAriaSort('last_observed_cycle')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Last Cycle</span>
                    {renderSortIcon('last_observed_cycle')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('actual_rul_official')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('actual_rul_official')}
                    aria-label={`Sort by Actual RUL, currently ${getAriaSort('actual_rul_official')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Actual RUL</span>
                    {renderSortIcon('actual_rul_official')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('predicted_rul_raw')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('predicted_rul_raw')}
                    aria-label={`Sort by Raw predicted RUL, currently ${getAriaSort('predicted_rul_raw')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Raw Pred</span>
                    {renderSortIcon('predicted_rul_raw')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('predicted_rul_final')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('predicted_rul_final')}
                    aria-label={`Sort by Final predicted RUL, currently ${getAriaSort('predicted_rul_final')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Final Pred</span>
                    {renderSortIcon('predicted_rul_final')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('prediction_error')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('prediction_error')}
                    aria-label={`Sort by Signed error, currently ${getAriaSort('prediction_error')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Signed Error</span>
                    {renderSortIcon('prediction_error')}
                  </button>
                </th>

                <th aria-sort={getAriaSort('absolute_error')} style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleSort('absolute_error')}
                    aria-label={`Sort by Absolute error, currently ${getAriaSort('absolute_error')}`}
                    style={{ justifyContent: 'flex-end', width: '100%' }}
                  >
                    <span>Abs Error</span>
                    {renderSortIcon('absolute_error')}
                  </button>
                </th>

                <th style={{ textAlign: 'center', width: '130px' }}>
                  <span>Class</span>
                </th>

                <th style={{ textAlign: 'center', width: '80px' }}>
                  <span>Action</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((row) => {
                const isSelected = row.unit_id === selectedUnitId;
                const isOver = row.prediction_error > 0;
                const isUnder = row.prediction_error < 0;

                return (
                  <tr
                    key={row.unit_id}
                    className={isSelected ? 'selected' : ''}
                    onClick={() => onSelectEngine(row.unit_id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)' }}>
                        <span className="measured" style={{ fontWeight: 500, color: 'var(--color-ink)' }}>
                          #{row.unit_id}
                        </span>
                        {isSelected && (
                          <span className="chip chip-accent" style={{ padding: '0 4px', fontSize: '0.65rem' }}>
                            selected
                          </span>
                        )}
                      </div>
                    </td>

                    <td className="measured" style={{ textAlign: 'right' }}>
                      {row.last_observed_cycle}
                    </td>

                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink)' }}>
                      {formatCycles(row.actual_rul_official)}
                    </td>

                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink-2)' }}>
                      {formatCycles(row.predicted_rul_raw)}
                    </td>

                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink)' }}>
                      {formatCycles(row.predicted_rul_final)}
                    </td>

                    <td
                      className="measured"
                      style={{
                        textAlign: 'right',
                        color: isOver ? 'var(--color-accent)' : isUnder ? 'var(--color-cool)' : 'var(--color-ink-2)',
                      }}
                    >
                      {formatSignedError(row.prediction_error)}
                    </td>

                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink-2)' }}>
                      {formatCycles(row.absolute_error)}
                    </td>

                    <td style={{ textAlign: 'center' }}>
                      <span
                        className={`chip ${isOver ? 'chip-accent' : isUnder ? 'chip-cool' : ''}`}
                        style={{ fontSize: '0.65rem' }}
                      >
                        {isOver ? '+ over (optimistic)' : isUnder ? '- under (conservative)' : 'exact'}
                      </span>
                    </td>

                    <td style={{ textAlign: 'center' }}>
                      <button
                        type="button"
                        className={`btn btn-sm ${isSelected ? 'btn-primary' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectEngine(row.unit_id);
                        }}
                        aria-label={`Inspect engine unit ${row.unit_id}`}
                        style={{ height: '24px', padding: '0 8px', fontSize: '0.7rem' }}
                      >
                        <Eye size={12} />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination Footer */}
      {sortedData.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
          <span>
            Showing {(safeCurrentPage - 1) * pageSize + 1} to{' '}
            {Math.min(safeCurrentPage * pageSize, sortedData.length)} of {sortedData.length} engines
          </span>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={safeCurrentPage === 1}
              aria-label="Previous page"
              style={{ opacity: safeCurrentPage === 1 ? 0.4 : 1 }}
            >
              <ChevronLeft size={14} />
              <span>Previous</span>
            </button>

            <span className="measured" style={{ padding: '0 var(--space-xs)' }}>
              Page {safeCurrentPage} of {totalPages}
            </span>

            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={safeCurrentPage === totalPages}
              aria-label="Next page"
              style={{ opacity: safeCurrentPage === totalPages ? 0.4 : 1 }}
            >
              <span>Next</span>
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
