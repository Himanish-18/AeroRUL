import React, { useState, useMemo } from 'react';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { PredictionRecord, SortField, SortDirection } from '../../lib/data/types';
import { formatCycles, formatSignedError } from '../../lib/utils/format';
import { filterAndSortPredictions } from '../../lib/data/adapter';
import { EmptyState } from '../../components/shared/EmptyState';

interface EnginePredictionTableProps {
  predictions: PredictionRecord[];
  selectedUnitId: number | null;
  onSelectEngine: (unitId: number) => void;
}

export const EnginePredictionTable: React.FC<EnginePredictionTableProps> = ({
  predictions,
  selectedUnitId,
  onSelectEngine,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('unit_id');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  // Toggle sort order
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
    setCurrentPage(1);
  };

  // Filter and sort
  const processedData = useMemo(() => {
    return filterAndSortPredictions(predictions, searchQuery, {
      field: sortField,
      direction: sortDirection,
    });
  }, [predictions, searchQuery, sortField, sortDirection]);

  // Pagination slice
  const totalPages = Math.ceil(processedData.length / pageSize) || 1;
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return processedData.slice(start, start + pageSize);
  }, [processedData, currentPage]);

  const renderSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown size={12} color="var(--color-rule-2)" />;
    }
    return sortDirection === 'asc' ? (
      <ArrowUp size={12} color="var(--color-accent)" />
    ) : (
      <ArrowDown size={12} color="var(--color-accent)" />
    );
  };

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Header and Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
        <div>
          <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
            Official Test Engine Predictions
          </h3>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Showing {processedData.length} engine trajectories · Click a row to inspect error breakdown
          </span>
        </div>

        {/* Search input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <div style={{ position: 'relative', width: '220px' }}>
            <Search
              size={14}
              color="var(--color-muted)"
              style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }}
            />
            <input
              type="text"
              className="input-text"
              style={{ width: '100%', paddingLeft: '32px' }}
              placeholder="Search by engine ID..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              aria-label="Search by engine ID"
            />
          </div>

          {searchQuery && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setSearchQuery('')}
              style={{ fontSize: 'var(--text-xs)' }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Table view */}
      {processedData.length === 0 ? (
        <EmptyState
          title="No engines match your query"
          description={`No engine ID matching "${searchQuery}" was found in the current dataset.`}
          actionText="Reset search filter"
          onAction={() => setSearchQuery('')}
        />
      ) : (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('unit_id')} style={{ cursor: 'pointer' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span>Engine ID</span>
                    {renderSortIcon('unit_id')}
                  </div>
                </th>
                <th onClick={() => handleSort('last_observed_cycle')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    <span>Last observed cycle</span>
                    {renderSortIcon('last_observed_cycle')}
                  </div>
                </th>
                <th onClick={() => handleSort('actual_rul_official')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    <span>Actual RUL</span>
                    {renderSortIcon('actual_rul_official')}
                  </div>
                </th>
                <th onClick={() => handleSort('predicted_rul_final')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    <span>Predicted RUL</span>
                    {renderSortIcon('predicted_rul_final')}
                  </div>
                </th>
                <th onClick={() => handleSort('prediction_error')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    <span>Signed error</span>
                    {renderSortIcon('prediction_error')}
                  </div>
                </th>
                <th onClick={() => handleSort('absolute_error')} style={{ cursor: 'pointer', textAlign: 'right' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    <span>Absolute error</span>
                    {renderSortIcon('absolute_error')}
                  </div>
                </th>
                <th style={{ textAlign: 'center', width: '100px' }}>
                  <span>Error type</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {paginatedData.map((row) => {
                const isSelected = row.unit_id === selectedUnitId;
                const isOver = row.prediction_error > 0;
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
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink)' }}>
                      {formatCycles(row.predicted_rul_final)}
                    </td>
                    <td
                      className="measured"
                      style={{
                        textAlign: 'right',
                        color: isOver ? 'var(--color-ok)' : 'var(--color-bad)',
                      }}
                    >
                      {formatSignedError(row.prediction_error)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-ink-2)' }}>
                      {formatCycles(row.absolute_error)}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <span className={`chip ${isOver ? 'chip-ok' : 'chip-bad'}`} style={{ fontSize: '0.7rem' }}>
                        {isOver ? '+ over' : '- under'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination toolbar */}
      {processedData.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
          <span>
            Showing {(currentPage - 1) * pageSize + 1} to{' '}
            {Math.min(currentPage * pageSize, processedData.length)} of {processedData.length} engines
          </span>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              aria-label="Previous page"
              style={{ opacity: currentPage === 1 ? 0.4 : 1 }}
            >
              <ChevronLeft size={14} />
              <span>Previous</span>
            </button>

            <span className="measured" style={{ padding: '0 var(--space-xs)' }}>
              Page {currentPage} of {totalPages}
            </span>

            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              aria-label="Next page"
              style={{ opacity: currentPage === totalPages ? 0.4 : 1 }}
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
