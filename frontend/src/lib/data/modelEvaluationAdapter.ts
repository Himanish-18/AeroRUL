/**
 * AeroRUL — Model Evaluation Data Adapter
 *
 * Pure query selectors, sorting functions, leader calculators, margin analyzers,
 * rank comparison, and sensitivity filtering for candidate model benchmarks.
 * Never mutates source records.
 */

import {
  DatasetId,
  Phase4BenchmarkRecord,
  Phase5CrossValidationRecord,
  Phase5DevelopmentValidationRecord,
  Phase5WindowAblationRecord,
  Phase5TargetCapSensitivityRecord,
  HyperparameterTrialRecord,
  Phase4SortMetric,
} from './types';

// ==========================================
// Phase 4 Benchmark Adapters
// ==========================================

export function getPhase4BenchmarkForDataset(
  records: Phase4BenchmarkRecord[],
  dataset: DatasetId
): Phase4BenchmarkRecord[] {
  return records.filter((r) => r.dataset === dataset);
}

export function sortPhase4Candidates(
  records: Phase4BenchmarkRecord[],
  metric: Phase4SortMetric = 'endpoint_mae',
  direction: 'asc' | 'desc' = 'asc'
): Phase4BenchmarkRecord[] {
  return [...records].sort((a, b) => {
    const valA = a[metric];
    const valB = b[metric];
    return direction === 'asc' ? valA - valB : valB - valA;
  });
}

export function getPhase4Leader(
  records: Phase4BenchmarkRecord[],
  metric: Phase4SortMetric = 'endpoint_mae'
): Phase4BenchmarkRecord | undefined {
  if (records.length === 0) return undefined;
  return sortPhase4Candidates(records, metric, 'asc')[0];
}

// ==========================================
// Phase 5 Cross-Validation Adapters
// ==========================================

export function getPhase5CvForDataset(
  records: Phase5CrossValidationRecord[],
  dataset: DatasetId
): Phase5CrossValidationRecord[] {
  return records.filter((r) => r.dataset === dataset);
}

export function sortPhase5CvCandidates(
  records: Phase5CrossValidationRecord[],
  direction: 'asc' | 'desc' = 'asc'
): Phase5CrossValidationRecord[] {
  return [...records].sort((a, b) => {
    return direction === 'asc'
      ? a.engine_macro_mae - b.engine_macro_mae
      : b.engine_macro_mae - a.engine_macro_mae;
  });
}

export function getPhase5CvLeader(
  records: Phase5CrossValidationRecord[]
): Phase5CrossValidationRecord | undefined {
  if (records.length === 0) return undefined;
  return sortPhase5CvCandidates(records, 'asc')[0];
}

// ==========================================
// Phase 5 Development Validation Adapters
// ==========================================

export function getPhase5DevValForDataset(
  records: Phase5DevelopmentValidationRecord[],
  dataset: DatasetId
): Phase5DevelopmentValidationRecord[] {
  return records.filter((r) => r.dataset === dataset);
}

export function sortPhase5DevValCandidates(
  records: Phase5DevelopmentValidationRecord[],
  direction: 'asc' | 'desc' = 'asc'
): Phase5DevelopmentValidationRecord[] {
  return [...records].sort((a, b) => {
    return direction === 'asc'
      ? a.engine_macro_mae - b.engine_macro_mae
      : b.engine_macro_mae - a.engine_macro_mae;
  });
}

export function getPhase5DevValLeader(
  records: Phase5DevelopmentValidationRecord[]
): Phase5DevelopmentValidationRecord | undefined {
  if (records.length === 0) return undefined;
  return sortPhase5DevValCandidates(records, 'asc')[0];
}

// ==========================================
// Cross-Validation vs Development-Validation Comparison
// ==========================================

export interface RankComparisonItem {
  model: string;
  cvRank: number;
  cvMae: number;
  devRank: number | null;
  devMae: number | null;
  rankDelta: number | null; // devRank - cvRank (positive = dropped rank, negative = improved rank)
  isReversal: boolean;
  isChallengerVariant: boolean;
}

export function compareCvAndDevValRanks(
  cvRecords: Phase5CrossValidationRecord[],
  devRecords: Phase5DevelopmentValidationRecord[]
): RankComparisonItem[] {
  const sortedCv = sortPhase5CvCandidates(cvRecords, 'asc');
  const sortedDev = sortPhase5DevValCandidates(devRecords, 'asc');

  const cvRankMap = new Map<string, { rank: number; mae: number }>();
  sortedCv.forEach((rec, idx) => {
    cvRankMap.set(rec.model, { rank: idx + 1, mae: rec.engine_macro_mae });
  });

  const devRankMap = new Map<string, { rank: number; mae: number }>();
  sortedDev.forEach((rec, idx) => {
    devRankMap.set(rec.model, { rank: idx + 1, mae: rec.engine_macro_mae });
  });

  // Group models, prioritizing models present in CV
  const allModels = Array.from(new Set([...sortedCv.map((r) => r.model), ...sortedDev.map((r) => r.model)]));

  return allModels.map((model) => {
    const cvInfo = cvRankMap.get(model);
    const devInfo = devRankMap.get(model);

    const cvRank = cvInfo ? cvInfo.rank : 999;
    const cvMae = cvInfo ? cvInfo.mae : 0;
    const devRank = devInfo ? devInfo.rank : null;
    const devMae = devInfo ? devInfo.mae : null;

    const rankDelta = devRank !== null && cvInfo ? devRank - cvRank : null;
    // A reversal occurs if the top CV model is not top in dev, or rank relative to direct competitor swapped
    const isReversal = rankDelta !== null && rankDelta !== 0;
    const isChallengerVariant = model.includes('Cap125');

    return {
      model,
      cvRank,
      cvMae,
      devRank,
      devMae,
      rankDelta,
      isReversal,
      isChallengerVariant,
    };
  }).sort((a, b) => a.cvRank - b.cvRank);
}

export interface SelectionMargin {
  selectedModel: string;
  selectedMae: number;
  runnerUpModel: string;
  runnerUpMae: number;
  absoluteMargin: number;
  percentageMargin: number;
  isNarrow: boolean; // < 0.1 cycle margin
}

export function calculateCvSelectionMargin(
  cvRecords: Phase5CrossValidationRecord[],
  selectedModelFamily: string
): SelectionMargin | null {
  const sorted = sortPhase5CvCandidates(cvRecords, 'asc');
  if (sorted.length < 2) return null;

  // Find candidate corresponding to selectedModelFamily (e.g. LightGBM -> LightGBM_Tuned)
  const selected = sorted.find((r) => r.model.toLowerCase().includes(selectedModelFamily.toLowerCase())) || sorted[0];
  const runnerUp = sorted.find((r) => r.model !== selected.model);

  if (!runnerUp) return null;

  const absoluteMargin = Math.abs(runnerUp.engine_macro_mae - selected.engine_macro_mae);
  const percentageMargin = selected.engine_macro_mae > 0
    ? (absoluteMargin / selected.engine_macro_mae) * 100
    : 0;

  return {
    selectedModel: selected.model,
    selectedMae: selected.engine_macro_mae,
    runnerUpModel: runnerUp.model,
    runnerUpMae: runnerUp.engine_macro_mae,
    absoluteMargin,
    percentageMargin,
    isNarrow: absoluteMargin < 0.1,
  };
}

// ==========================================
// Sensitivity Adapters
// ==========================================

export function getWindowAblationForDataset(
  records: Phase5WindowAblationRecord[],
  dataset: DatasetId
): Phase5WindowAblationRecord[] {
  return records.filter((r) => r.dataset === dataset);
}

export function getTargetCapSensitivityForDataset(
  records: Phase5TargetCapSensitivityRecord[],
  dataset: DatasetId
): Phase5TargetCapSensitivityRecord[] {
  return records.filter((r) => r.dataset === dataset);
}

// ==========================================
// Hyperparameter Trial Adapters
// ==========================================

export function getHyperparameterTrialsForDataset(
  records: HyperparameterTrialRecord[],
  dataset: DatasetId,
  modelFamily?: string
): HyperparameterTrialRecord[] {
  let filtered = records.filter((r) => r.dataset === dataset);
  if (modelFamily && modelFamily !== 'all') {
    filtered = filtered.filter((r) => r.model_family.toLowerCase() === modelFamily.toLowerCase());
  }
  return [...filtered].sort((a, b) => a.trial_id - b.trial_id);
}

export function getBestHyperparameterTrial(
  trials: HyperparameterTrialRecord[]
): HyperparameterTrialRecord | undefined {
  const completed = trials.filter((t) => t.status === 'COMPLETE');
  if (completed.length === 0) return undefined;
  return [...completed].sort((a, b) => a.engine_macro_mae - b.engine_macro_mae)[0];
}

export interface HyperparameterTrialSummary {
  totalStoredTrials: number;
  completedTrials: number;
  bestEngineMacroMae: number;
  medianEngineMacroMae: number;
  bestPooledMae: number;
  bestPooledRmse: number;
  bestTrialId: number;
  runtimeSecBestTrial: number;
}

export function getHyperparameterTrialSummary(
  trials: HyperparameterTrialRecord[]
): HyperparameterTrialSummary | null {
  if (trials.length === 0) return null;

  const completed = trials.filter((t) => t.status === 'COMPLETE');
  if (completed.length === 0) return null;

  const sortedByMae = [...completed].sort((a, b) => a.engine_macro_mae - b.engine_macro_mae);
  const best = sortedByMae[0];

  const mid = Math.floor(sortedByMae.length / 2);
  const medianEngineMacroMae = sortedByMae.length % 2 !== 0
    ? sortedByMae[mid].engine_macro_mae
    : (sortedByMae[mid - 1].engine_macro_mae + sortedByMae[mid].engine_macro_mae) / 2;

  const bestPooledMae = [...completed].sort((a, b) => a.pooled_mae - b.pooled_mae)[0].pooled_mae;
  const bestPooledRmse = [...completed].sort((a, b) => a.pooled_rmse - b.pooled_rmse)[0].pooled_rmse;

  return {
    totalStoredTrials: trials.length,
    completedTrials: completed.length,
    bestEngineMacroMae: best.engine_macro_mae,
    medianEngineMacroMae,
    bestPooledMae,
    bestPooledRmse,
    bestTrialId: best.trial_id,
    runtimeSecBestTrial: best.runtime_sec,
  };
}
