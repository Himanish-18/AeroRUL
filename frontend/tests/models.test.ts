import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  parsePhase4BenchmarkCsv,
  parsePhase5CrossValidationCsv,
  parsePhase5DevelopmentValidationCsv,
  parsePhase5WindowAblationCsv,
  parsePhase5TargetCapSensitivityCsv,
  parseHyperparameterSearchCsv,
  parseModelManifest,
} from '../src/lib/data/parser';
import {
  getPhase4BenchmarkForDataset,
  sortPhase4Candidates,
  getPhase4Leader,
  getPhase5CvForDataset,
  sortPhase5CvCandidates,
  getPhase5CvLeader,
  getPhase5DevValForDataset,
  sortPhase5DevValCandidates,
  compareCvAndDevValRanks,
  calculateCvSelectionMargin,
  getWindowAblationForDataset,
  getTargetCapSensitivityForDataset,
  getHyperparameterTrialsForDataset,
  getBestHyperparameterTrial,
  getHyperparameterTrialSummary,
} from '../src/lib/data/modelEvaluationAdapter';
import {
  Phase4BenchmarkRecord,
  Phase5CrossValidationRecord,
  Phase5DevelopmentValidationRecord,
  Phase5WindowAblationRecord,
  Phase5TargetCapSensitivityRecord,
  HyperparameterTrialRecord,
  ModelManifest,
  VALID_DATASETS,
} from '../src/lib/data/types';

describe('AeroRUL Stage 3 — Model Evaluation & Benchmark Analysis', () => {
  let p4CsvRaw: string;
  let p5CvCsvRaw: string;
  let p5DevValCsvRaw: string;
  let p5WindowCsvRaw: string;
  let p5CapCsvRaw: string;
  let p5HyperCsvRaw: string;
  let manifestJsonRaw: any;

  let p4Records: Phase4BenchmarkRecord[];
  let p5CvRecords: Phase5CrossValidationRecord[];
  let p5DevValRecords: Phase5DevelopmentValidationRecord[];
  let p5WindowRecords: Phase5WindowAblationRecord[];
  let p5CapRecords: Phase5TargetCapSensitivityRecord[];
  let p5HyperRecords: HyperparameterTrialRecord[];
  let manifest: ModelManifest;

  beforeAll(() => {
    const p4Dir = path.resolve(__dirname, '../public/data/phase4');
    const p5Dir = path.resolve(__dirname, '../public/data/phase5');

    p4CsvRaw = fs.readFileSync(path.join(p4Dir, 'model_benchmark_metrics.csv'), 'utf-8');
    p5CvCsvRaw = fs.readFileSync(path.join(p5Dir, 'cross_validation_metrics.csv'), 'utf-8');
    p5DevValCsvRaw = fs.readFileSync(path.join(p5Dir, 'development_validation_metrics.csv'), 'utf-8');
    p5WindowCsvRaw = fs.readFileSync(path.join(p5Dir, 'window_ablation_metrics.csv'), 'utf-8');
    p5CapCsvRaw = fs.readFileSync(path.join(p5Dir, 'target_cap_sensitivity.csv'), 'utf-8');
    p5HyperCsvRaw = fs.readFileSync(path.join(p5Dir, 'hyperparameter_search_results.csv'), 'utf-8');
    manifestJsonRaw = JSON.parse(fs.readFileSync(path.join(p5Dir, 'final_model_manifest.json'), 'utf-8'));

    p4Records = parsePhase4BenchmarkCsv(p4CsvRaw);
    p5CvRecords = parsePhase5CrossValidationCsv(p5CvCsvRaw);
    p5DevValRecords = parsePhase5DevelopmentValidationCsv(p5DevValCsvRaw);
    p5WindowRecords = parsePhase5WindowAblationCsv(p5WindowCsvRaw);
    p5CapRecords = parsePhase5TargetCapSensitivityCsv(p5CapCsvRaw);
    p5HyperRecords = parseHyperparameterSearchCsv(p5HyperCsvRaw);
    manifest = parseModelManifest(manifestJsonRaw);
  });

  // ==========================================
  // Section A: Parser Tests
  // ==========================================
  describe('A. Parser Functionality & Edge Cases', () => {
    it('throws error when parsing empty or whitespace-only CSV', () => {
      expect(() => parsePhase4BenchmarkCsv('')).toThrow('empty or has no data rows');
      expect(() => parsePhase5CrossValidationCsv('   \n  ')).toThrow('empty or has no data rows');
      expect(() => parsePhase5DevelopmentValidationCsv('')).toThrow('empty or has no data rows');
      expect(() => parsePhase5WindowAblationCsv('')).toThrow('empty or has no data rows');
      expect(() => parsePhase5TargetCapSensitivityCsv('')).toThrow('empty or has no data rows');
      expect(() => parseHyperparameterSearchCsv('')).toThrow('empty or has no data rows');
    });

    it('throws descriptive error on invalid dataset identifier', () => {
      const badCsv = 'dataset,model,row_mae,row_rmse,row_r2,n_observations,macro_mae,macro_rmse,n_engines,endpoint_mae,endpoint_rmse,endpoint_r2,endpoint_nasa_score,endpoint_nasa_avg,mean_signed_error,n_underpredictions,n_overpredictions,runtime\nFD999,Ridge,10,12,0.8,100,10,12,20,15,18,0,50,2.5,-1,10,10,1.2';
      expect(() => parsePhase4BenchmarkCsv(badCsv)).toThrow(/Invalid dataset identifier/);
    });

    it('throws descriptive error on malformed non-finite float', () => {
      const badCsv = 'dataset,model,engine_macro_mae,engine_macro_mae_std,engine_macro_rmse,pooled_mae,pooled_rmse,mean_signed_error,nasa_score_sum,nasa_score_mean\nFD001,LightGBM,NaN_INVALID,0.5,15,12,16,-0.5,100,10';
      expect(() => parsePhase5CrossValidationCsv(badCsv)).toThrow(/Invalid non-finite float/);
    });

    it('correctly parses JSON in params field of hyperparameter search CSV', () => {
      const firstTrial = p5HyperRecords[0];
      expect(typeof firstTrial.params).toBe('object');
      expect(firstTrial.params).not.toBeNull();
      expect(firstTrial.params_raw).toBeTruthy();
      expect(firstTrial.trial_id).toBe(0);
    });

    it('throws descriptive error on invalid JSON in hyperparameter params', () => {
      const badCsv = 'dataset,model_family,trial_id,params,engine_macro_mae,pooled_mae,pooled_rmse,nasa_score_mean,runtime_sec,status\nFD001,LightGBM,0,{not_json},15,14,20,100,10.5,COMPLETE';
      expect(() => parseHyperparameterSearchCsv(badCsv)).toThrow(/Invalid JSON in params/);
    });

    it('correctly parses target_cap as None or numeric', () => {
      const fd001CapNone = p5CapRecords.find((r) => r.dataset === 'FD001' && r.target_cap === 'None');
      const fd001Cap150 = p5CapRecords.find((r) => r.dataset === 'FD001' && r.target_cap === 150);
      expect(fd001CapNone).toBeDefined();
      expect(fd001Cap150).toBeDefined();
      expect(fd001Cap150?.target_cap).toBe(150);
    });
  });

  // ==========================================
  // Section B: Data Integrity & Inventory
  // ==========================================
  describe('B. Artifact Inventory & Exact Record Counts', () => {
    it('Phase 4 benchmark contains exactly 24 records (6 models per dataset)', () => {
      expect(p4Records).toHaveLength(24);
      for (const ds of VALID_DATASETS) {
        const dsRecords = p4Records.filter((r) => r.dataset === ds);
        expect(dsRecords).toHaveLength(6);
        const models = dsRecords.map((r) => r.model).sort();
        expect(models).toEqual([
          'DummyBaseline',
          'LightGBM',
          'LinearRegression',
          'RandomForest',
          'RidgeRegression',
          'XGBoost',
        ]);
      }
    });

    it('Phase 5 CV contains exactly 20 records (5 models per dataset)', () => {
      expect(p5CvRecords).toHaveLength(20);
      for (const ds of VALID_DATASETS) {
        const dsRecords = p5CvRecords.filter((r) => r.dataset === ds);
        expect(dsRecords).toHaveLength(5);
        const models = dsRecords.map((r) => r.model).sort();
        expect(models).toEqual([
          'DummyBaseline',
          'LightGBM_Tuned',
          'RandomForest_Fixed',
          'RidgeRegression',
          'XGBoost_Tuned',
        ]);
      }
    });

    it('Phase 5 development validation contains exactly 24 records (6 models per dataset)', () => {
      expect(p5DevValRecords).toHaveLength(24);
      for (const ds of VALID_DATASETS) {
        const dsRecords = p5DevValRecords.filter((r) => r.dataset === ds);
        expect(dsRecords).toHaveLength(6);
        const models = dsRecords.map((r) => r.model).sort();
        expect(models).toEqual([
          'DummyBaseline',
          'LightGBM_Tuned',
          'LightGBM_Tuned_Cap125',
          'RandomForest_Fixed',
          'RidgeRegression',
          'XGBoost_Tuned',
        ]);
      }
    });

    it('Window ablation contains exactly 24 records (6 window configs per dataset)', () => {
      expect(p5WindowRecords).toHaveLength(24);
      for (const ds of VALID_DATASETS) {
        const dsRecords = p5WindowRecords.filter((r) => r.dataset === ds);
        expect(dsRecords).toHaveLength(6);
        const configs = dsRecords.map((r) => r.window_configuration).sort();
        expect(configs).toEqual(['ALL_WINDOWS', 'W10', 'W20', 'W30', 'W50', 'W75']);
      }
    });

    it('Target cap sensitivity contains exactly 24 records (6 caps per dataset)', () => {
      expect(p5CapRecords).toHaveLength(24);
      for (const ds of VALID_DATASETS) {
        const dsRecords = p5CapRecords.filter((r) => r.dataset === ds);
        expect(dsRecords).toHaveLength(6);
      }
    });

    it('Hyperparameter search contains exactly 100 stored trials (15 LightGBM, 10 XGBoost per dataset)', () => {
      expect(p5HyperRecords).toHaveLength(100);
      for (const ds of VALID_DATASETS) {
        const dsTrials = p5HyperRecords.filter((r) => r.dataset === ds);
        expect(dsTrials).toHaveLength(25);
        const lgbTrials = dsTrials.filter((r) => r.model_family === 'LightGBM');
        const xgbTrials = dsTrials.filter((r) => r.model_family === 'XGBoost');
        expect(lgbTrials).toHaveLength(15);
        expect(xgbTrials).toHaveLength(10);
      }
    });
  });

  // ==========================================
  // Section C: Model Comparison & Scientific Rationale
  // ==========================================
  describe('C. Model Comparison & Scientific Rationale', () => {
    it('Phase 4 ranking by endpoint MAE orders models ascending with non-mutating sort', () => {
      const fd001P4 = getPhase4BenchmarkForDataset(p4Records, 'FD001');
      const originalFirst = fd001P4[0].model;
      const sorted = sortPhase4Candidates(fd001P4, 'endpoint_mae', 'asc');
      expect(fd001P4[0].model).toBe(originalFirst); // original not mutated

      const leader = getPhase4Leader(fd001P4, 'endpoint_mae');
      expect(leader).toBeDefined();
      expect(leader?.model).toBe(sorted[0].model);

      for (let i = 0; i < sorted.length - 1; i++) {
        expect(sorted[i].endpoint_mae).toBeLessThanOrEqual(sorted[i + 1].endpoint_mae);
      }
    });

    it('Phase 4 alternate ranking by row MAE functions correctly', () => {
      const fd001P4 = getPhase4BenchmarkForDataset(p4Records, 'FD001');
      const sorted = sortPhase4Candidates(fd001P4, 'row_mae', 'asc');
      for (let i = 0; i < sorted.length - 1; i++) {
        expect(sorted[i].row_mae).toBeLessThanOrEqual(sorted[i + 1].row_mae);
      }
    });

    it('Phase 5 CV ranking by engine_macro_mae identifies correct winner', () => {
      for (const ds of VALID_DATASETS) {
        const dsCv = getPhase5CvForDataset(p5CvRecords, ds);
        const leader = getPhase5CvLeader(dsCv);
        expect(leader).toBeDefined();
        expect(leader?.model).toBe('LightGBM_Tuned');
      }
    });

    it('Frozen model manifest lookup matches exact frozen configurations', () => {
      expect(manifest.selected_configurations.FD001.model_family).toBe('LightGBM');
      expect(manifest.selected_configurations.FD001.feature_configuration).toBe('ALL_WINDOWS');
      expect(manifest.selected_configurations.FD001.target_cap).toBe('150');

      expect(manifest.selected_configurations.FD002.model_family).toBe('LightGBM');
      expect(manifest.selected_configurations.FD002.feature_configuration).toBe('W75');
      expect(manifest.selected_configurations.FD002.target_cap).toBe('150');

      expect(manifest.selected_configurations.FD003.model_family).toBe('LightGBM');
      expect(manifest.selected_configurations.FD003.feature_configuration).toBe('ALL_WINDOWS');
      expect(manifest.selected_configurations.FD003.target_cap).toBe('None');

      expect(manifest.selected_configurations.FD004.model_family).toBe('LightGBM');
      expect(manifest.selected_configurations.FD004.feature_configuration).toBe('ALL_WINDOWS');
      expect(manifest.selected_configurations.FD004.target_cap).toBe('None');
    });

    it('calculates positive CV selection margins for LightGBM across all subsets', () => {
      for (const ds of VALID_DATASETS) {
        const dsCv = getPhase5CvForDataset(p5CvRecords, ds);
        const margin = calculateCvSelectionMargin(dsCv, 'LightGBM');
        expect(margin).toBeDefined();
        expect(margin?.selectedModel).toBe('LightGBM_Tuned');
        expect(margin?.runnerUpModel).toBe('XGBoost_Tuned');
        expect(margin?.absoluteMargin).toBeGreaterThan(0);
      }
    });

    it('accurately detects the FD004 CV vs Dev Val rank reversal with near-tie margin', () => {
      const fd004Cv = getPhase5CvForDataset(p5CvRecords, 'FD004');
      const fd004Dev = getPhase5DevValForDataset(p5DevValRecords, 'FD004');

      // Inner CV: LightGBM won narrowly
      const cvSorted = sortPhase5CvCandidates(fd004Cv, 'asc');
      expect(cvSorted[0].model).toBe('LightGBM_Tuned');
      expect(cvSorted[1].model).toBe('XGBoost_Tuned');
      expect(cvSorted[0].engine_macro_mae).toBeCloseTo(25.5999, 3);
      expect(cvSorted[1].engine_macro_mae).toBeCloseTo(25.6254, 3);

      const margin = calculateCvSelectionMargin(fd004Cv, 'LightGBM');
      expect(margin?.isNarrow).toBe(true);
      expect(margin?.absoluteMargin).toBeCloseTo(0.0255, 3);

      // Development Validation: XGBoost won narrowly (rank reversal!)
      const devSorted = sortPhase5DevValCandidates(fd004Dev, 'asc');
      expect(devSorted[0].model).toBe('XGBoost_Tuned');
      expect(devSorted[1].model).toBe('LightGBM_Tuned');
      expect(devSorted[0].engine_macro_mae).toBeCloseTo(27.5943, 3);
      expect(devSorted[1].engine_macro_mae).toBeCloseTo(27.6324, 3);

      // Compare rank structure
      const comp = compareCvAndDevValRanks(fd004Cv, fd004Dev);
      const lgbComp = comp.find((c) => c.model === 'LightGBM_Tuned');
      const xgbComp = comp.find((c) => c.model === 'XGBoost_Tuned');
      expect(lgbComp?.cvRank).toBe(1);
      expect(lgbComp?.devRank).toBe(2);
      expect(xgbComp?.cvRank).toBe(2);
      expect(xgbComp?.devRank).toBe(1);
      expect(lgbComp?.isReversal).toBe(true);
    });

    it('correctly filters window ablation and target cap records by dataset without cross-contamination', () => {
      const fd002Windows = getWindowAblationForDataset(p5WindowRecords, 'FD002');
      expect(fd002Windows).toHaveLength(6);
      expect(fd002Windows.every((r) => r.dataset === 'FD002')).toBe(true);

      const fd003Caps = getTargetCapSensitivityForDataset(p5CapRecords, 'FD003');
      expect(fd003Caps).toHaveLength(6);
      expect(fd003Caps.every((r) => r.dataset === 'FD003')).toBe(true);
    });

    it('identifies best stored Optuna trial per dataset and model family', () => {
      const fd001Trials = getHyperparameterTrialsForDataset(p5HyperRecords, 'FD001', 'LightGBM');
      expect(fd001Trials).toHaveLength(15);

      const best = getBestHyperparameterTrial(fd001Trials);
      expect(best).toBeDefined();
      expect(best?.model_family).toBe('LightGBM');

      const summary = getHyperparameterTrialSummary(fd001Trials);
      expect(summary).toBeDefined();
      expect(summary?.completedTrials).toBe(15);
      expect(summary?.bestEngineMacroMae).toBe(best?.engine_macro_mae);
      expect(summary?.bestTrialId).toBe(best?.trial_id);
    });
  });
});
