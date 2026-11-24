/**
 * AeroRUL — Numerical and Engineering Measurement Formatting
 *
 * All functions return strings with strict tabular alignment and unit designations.
 * Ensures consistent precision across dashboard tiles, tables, and tooltips.
 */

export function formatCycles(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(decimals);
}

export function formatSignedError(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)}`;
}

export function formatR2(value: number | undefined | null, decimals = 4): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(decimals);
}

export function formatNasaScore(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  if (value >= 1000) {
    return Math.round(value).toLocaleString('en-US');
  }
  return value.toFixed(1);
}

export function formatInteger(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  return Math.round(value).toLocaleString('en-US');
}

export function formatPercent(value: number | undefined | null, decimals = 1): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatSignedDelta(value: number | undefined | null, decimals = 2): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  if (Math.abs(value) < 0.0001) {
    return '0.00';
  }
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(decimals)}`;
}

export function formatRuntime(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—';
  }
  if (value < 0.01) {
    return '< 0.01s';
  }
  if (value >= 60) {
    const mins = Math.floor(value / 60);
    const secs = (value % 60).toFixed(1);
    return `${mins}m ${secs}s`;
  }
  return `${value.toFixed(2)}s`;
}



