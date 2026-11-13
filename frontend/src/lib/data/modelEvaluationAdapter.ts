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
  RankComparisonItem,
  HyperparameterTrialSummary,
} from './types';

export type { RankComparisonItem, HyperparameterTrialSummary };

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

export function compareCvAndDevValRanks(
  cvRecords: Phase5CrossValidationRecord[],
  devRecords: Phase5DevelopmentValidationRecord[]
): RankComparisonItem[] {
  // 1. Sort CV candidates (5 common models in CV)
  const sortedCv = sortPhase5CvCandidates(cvRecords, 'asc');
  const cvModelSet = new Set(sortedCv.map((r) => r.model));

  const cvRankMap = new Map<string, { rank: number; mae: number }>();
  sortedCv.forEach((rec, idx) => {
    cvRankMap.set(rec.model, { rank: idx + 1, mae: rec.engine_macro_mae });
  });

  // 2. Sort ALL dev records (6 models: includes Cap125 challenger)
  const sortedDevOverall = sortPhase5DevValCandidates(devRecords, 'asc');
  const devOverallRankMap = new Map<string, { rank: number; mae: number }>();
  sortedDevOverall.forEach((rec, idx) => {
    devOverallRankMap.set(rec.model, { rank: idx + 1, mae: rec.engine_macro_mae });
  });

  // 3. Calculate ranks for common candidate set (excluding Cap125)
  const commonDevRecords = sortedDevOverall.filter((r) => cvModelSet.has(r.model));
  const sortedCommonDev = sortPhase5DevValCandidates(commonDevRecords, 'asc');
  const devCommonRankMap = new Map<string, { rank: number; mae: number }>();
  sortedCommonDev.forEach((rec, idx) => {
    devCommonRankMap.set(rec.model, { rank: idx + 1, mae: rec.engine_macro_mae });
  });

  // 4. Map each candidate model
  const allModels = Array.from(new Set([...sortedDevOverall.map((r) => r.model)]));

  return allModels.map((model) => {
    const cvInfo = cvRankMap.get(model);
    const devOverallInfo = devOverallRankMap.get(model);
    const devCommonInfo = devCommonRankMap.get(model);
    const isChallengerVariant = model.includes('Cap125') || !cvInfo;

    const cvRank = cvInfo ? cvInfo.rank : null;
    const cvMae = cvInfo ? cvInfo.mae : null;
    const devRank = devOverallInfo ? devOverallInfo.rank : null;
    const devCommonRank = devCommonInfo ? devCommonInfo.rank : null;
    const devMae = devOverallInfo ? devOverallInfo.mae : null;

    // Rank delta and reversal are defined strictly over the shared candidate set
    const rankDelta = (cvRank !== null && devCommonRank !== null) ? devCommonRank - cvRank : null;
    const isReversal = cvRank !== null && devCommonRank !== null && cvRank !== devCommonRank;

    return {
      model,
      cvRank,
      cvMae,
      devRank,
      devCommonRank,
      devMae,
      rankDelta,
      isReversal,
      isChallengerVariant,
    };
  }).sort((a, b) => {
    if (a.cvRank !== null && b.cvRank !== null) return a.cvRank - b.cvRank;
    if (a.cvRank !== null) return -1;
    if (b.cvRank !== null) return 1;
    return (a.devRank || 999) - (b.devRank || 999);
  });
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

export function getTrialIdentity(
  dataset: DatasetId,
  modelFamily: string,
  trialId: number
): string {
  return `${dataset}:${modelFamily}:${trialId}`;
}

export function parseTrialIdentity(identity: string): {
  dataset: DatasetId;
  modelFamily: string;
  trialId: number;
} | null {
  const parts = identity.split(':');
  if (parts.length !== 3) return null;
  return {
    dataset: parts[0] as DatasetId,
    modelFamily: parts[1],
    trialId: parseInt(parts[2], 10),
  };
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

  const bestPooledMaeTrial = [...completed].sort((a, b) => a.pooled_mae - b.pooled_mae)[0];
  const bestPooledRmseTrial = [...completed].sort((a, b) => a.pooled_rmse - b.pooled_rmse)[0];

  return {
    totalStoredTrials: trials.length,
    completedTrials: completed.length,
    bestEngineMacroMae: best.engine_macro_mae,
    medianEngineMacroMae,
    bestPooledMae: bestPooledMaeTrial.pooled_mae,
    bestPooledMaeTrialId: bestPooledMaeTrial.trial_id,
    bestPooledRmse: bestPooledRmseTrial.pooled_rmse,
    bestPooledRmseTrialId: bestPooledRmseTrial.trial_id,
    bestTrialId: best.trial_id,
    runtimeSecBestTrial: best.runtime_sec,
  };
}

