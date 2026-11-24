import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  parseFeatureImportanceCsv,
  parseSensorImportanceCsv,
  parseFeatureFamilyCsv,
  parseSensorTrajectoriesCsv,
} from '../src/lib/data/parser';
import {
  getFilteredFeatures,
  getCorrelationComparisonList,
  getTrajectoryChartData,
  getEngineSensorStatistics,
  getSensorSpec,
} from '../src/lib/data/featureIntelligenceAdapter';
import {
  FeatureImportanceRecord,
  SensorTrajectoryRecord,
} from '../src/lib/data/types';

describe('Stage 4 — Feature Intelligence & Degradation Data Layer', () => {
  describe('1. Feature Importance CSV Parser', () => {
    const validCsv = `dataset,feature_name,sensor,statistic,window_or_lag,description,gain_importance,split_importance,gain_importance_pct,split_importance_pct,gain_rank,split_rank,correlation_pearson,correlation_spearman,abs_pearson,abs_spearman
FD001,sensor_11_max_10,sensor_11,max,10,Rolling maximum over 10 cycles,1000.5,40,25.0,20.0,1,2,-0.7231,-0.7838,0.7231,0.7838
FD001,sensor_4_mean_10,sensor_4,mean,10,Rolling mean over 10 cycles,800.0,50,20.0,25.0,2,1,,,,`;

    it('parses valid CSV and distinguishes missing correlations from zero', () => {
      const records = parseFeatureImportanceCsv(validCsv, 'FD001');
      expect(records).toHaveLength(2);

      const r1 = records[0];
      expect(r1.dataset).toBe('FD001');
      expect(r1.feature_name).toBe('sensor_11_max_10');
      expect(r1.gain_importance).toBe(1000.5);
      expect(r1.split_importance).toBe(40);
      expect(r1.gain_rank).toBe(1);
      expect(r1.split_rank).toBe(2);
      expect(r1.correlation_pearson).toBe(-0.7231);
      expect(r1.abs_pearson).toBe(0.7231);

      const r2 = records[1];
      expect(r2.correlation_pearson).toBeNull();
      expect(r2.abs_pearson).toBeNull();
      expect(r2.correlation_spearman).toBeNull();
      expect(r2.abs_spearman).toBeNull();
    });

    it('throws on missing required headers', () => {
      const invalidHeaderCsv = `dataset,feature_name,gain_importance\nFD001,feat_1,10.0`;
      expect(() => parseFeatureImportanceCsv(invalidHeaderCsv, 'FD001')).toThrow(
        /Missing required headers/
      );
    });

    it('throws on duplicate feature records', () => {
      const dupCsv = `dataset,feature_name,sensor,statistic,window_or_lag,description,gain_importance,split_importance,gain_importance_pct,split_importance_pct,gain_rank,split_rank,correlation_pearson,correlation_spearman,abs_pearson,abs_spearman
FD001,feat_dup,sensor_1,mean,10,desc,10.0,1,50.0,50.0,1,1,,,,
FD001,feat_dup,sensor_1,mean,10,desc,20.0,2,50.0,50.0,2,2,,,,`;
      expect(() => parseFeatureImportanceCsv(dupCsv, 'FD001')).toThrow(/Duplicate feature/);
    });

    it('throws on negative gain importance', () => {
      const negCsv = `dataset,feature_name,sensor,statistic,window_or_lag,description,gain_importance,split_importance,gain_importance_pct,split_importance_pct,gain_rank,split_rank,correlation_pearson,correlation_spearman,abs_pearson,abs_spearman
FD001,feat_1,sensor_1,mean,10,desc,-5.0,1,50.0,50.0,1,1,,,,`;
      expect(() => parseFeatureImportanceCsv(negCsv, 'FD001')).toThrow(
        /cannot be negative/
      );
    });
  });

  describe('2. Sensor Importance CSV Parser', () => {
    const validSensorCsv = `dataset,sensor,gain_importance,split_importance,gain_importance_pct,split_importance_pct,gain_rank,split_rank,feature_count,top_feature
FD001,sensor_11,5000.0,200,45.5,35.0,1,1,32,sensor_11_r2_75
FD001,sensor_4,3000.0,150,27.3,26.2,2,2,30,sensor_4_mean_10`;

    it('parses sensor importance records correctly', () => {
      const records = parseSensorImportanceCsv(validSensorCsv, 'FD001');
      expect(records).toHaveLength(2);
      expect(records[0].sensor).toBe('sensor_11');
      expect(records[0].gain_importance).toBe(5000.0);
      expect(records[0].split_importance).toBe(200);
      expect(records[0].feature_count).toBe(32);
      expect(records[0].top_feature).toBe('sensor_11_r2_75');
    });

    it('detects duplicate sensor records', () => {
      const dupCsv = `dataset,sensor,gain_importance,split_importance,gain_importance_pct,split_importance_pct,gain_rank,split_rank,feature_count,top_feature
FD001,sensor_11,5000.0,200,45.5,35.0,1,1,32,sensor_11_r2_75
FD001,sensor_11,3000.0,150,27.3,26.2,2,2,30,sensor_11_r2_75`;
      expect(() => parseSensorImportanceCsv(dupCsv, 'FD001')).toThrow(/Duplicate sensor/);
    });
  });

  describe('3. Feature Family CSV Parser', () => {
    const validFamilyCsv = `dataset,family_type,family_key,gain_importance,split_importance,gain_importance_pct,split_importance_pct,feature_count,rank_in_family
FD001,sensor,sensor_11,5000.0,200,45.5,35.0,32,1
FD001,statistic,mean,4000.0,180,36.4,31.5,150,1
FD001,window,W10,3500.0,160,31.8,28.0,120,1`;

    it('parses feature family records correctly', () => {
      const records = parseFeatureFamilyCsv(validFamilyCsv);
      expect(records).toHaveLength(3);
      expect(records[0].family_type).toBe('sensor');
      expect(records[1].family_type).toBe('statistic');
      expect(records[2].family_type).toBe('window');
    });

    it('rejects invalid family_type enum', () => {
      const invalidCsv = `dataset,family_type,family_key,gain_importance,split_importance,gain_importance_pct,split_importance_pct,feature_count,rank_in_family
FD001,unknown_type,sensor_11,5000.0,200,45.5,35.0,32,1`;
      expect(() => parseFeatureFamilyCsv(invalidCsv)).toThrow(/Invalid family_type/);
    });
  });

  describe('4. Sensor Trajectories CSV Parser', () => {
    const validTrajCsv = `dataset,unit_id,cycle,sensor,sensor_value
FD001,1,1,sensor_11,47.5412
FD001,1,3,sensor_11,47.6105
FD001,1,1,sensor_4,1408.25`;

    it('parses sensor trajectories with numeric checks', () => {
      const records = parseSensorTrajectoriesCsv(validTrajCsv, 'FD001');
      expect(records).toHaveLength(3);
      expect(records[0].unit_id).toBe(1);
      expect(records[0].cycle).toBe(1);
      expect(records[0].sensor).toBe('sensor_11');
      expect(records[0].sensor_value).toBe(47.5412);
    });

    it('rejects duplicate trajectory records for same unit, cycle, and sensor', () => {
      const dupCsv = `dataset,unit_id,cycle,sensor,sensor_value
FD001,1,1,sensor_11,47.5412
FD001,1,1,sensor_11,47.6000`;
      expect(() => parseSensorTrajectoriesCsv(dupCsv, 'FD001')).toThrow(
        /Duplicate trajectory point/
      );
    });

    it('rejects non-positive cycles', () => {
      const invalidCycleCsv = `dataset,unit_id,cycle,sensor,sensor_value
FD001,1,0,sensor_11,47.5412`;
      expect(() => parseSensorTrajectoriesCsv(invalidCycleCsv, 'FD001')).toThrow(
        /Invalid positive integer/
      );
    });
  });

  describe('5. Adapter Functionality', () => {
    const sampleFeatures: FeatureImportanceRecord[] = [
      {
        dataset: 'FD001',
        feature_name: 'sensor_11_r2_75',
        sensor: 'sensor_11',
        statistic: 'r2',
        window_or_lag: '75',
        description: 'r2 of sensor 11',
        gain_importance: 5000,
        split_importance: 20,
        gain_importance_pct: 50.0,
        split_importance_pct: 20.0,
        gain_rank: 1,
        split_rank: 2,
        correlation_pearson: -0.75,
        correlation_spearman: -0.80,
        abs_pearson: 0.75,
        abs_spearman: 0.80,
      },
      {
        dataset: 'FD001',
        feature_name: 'sensor_4_mean_10',
        sensor: 'sensor_4',
        statistic: 'mean',
        window_or_lag: '10',
        description: 'mean of sensor 4',
        gain_importance: 3000,
        split_importance: 50,
        gain_importance_pct: 30.0,
        split_importance_pct: 50.0,
        gain_rank: 2,
        split_rank: 1,
        correlation_pearson: 0.65,
        correlation_spearman: 0.70,
        abs_pearson: 0.65,
        abs_spearman: 0.70,
      },
      {
        dataset: 'FD001',
        feature_name: 'sensor_15_std_20',
        sensor: 'sensor_15',
        statistic: 'std',
        window_or_lag: '20',
        description: 'std of sensor 15',
        gain_importance: 2000,
        split_importance: 30,
        gain_importance_pct: 20.0,
        split_importance_pct: 30.0,
        gain_rank: 3,
        split_rank: 3,
        correlation_pearson: null,
        correlation_spearman: null,
        abs_pearson: null,
        abs_spearman: null,
      },
    ];

    it('filters and sorts features by gain vs split', () => {
      const byGain = getFilteredFeatures(sampleFeatures, 'gain', 'all');
      expect(byGain[0].feature_name).toBe('sensor_11_r2_75');
      expect(byGain[1].feature_name).toBe('sensor_4_mean_10');

      const bySplit = getFilteredFeatures(sampleFeatures, 'split', 'all');
      expect(bySplit[0].feature_name).toBe('sensor_4_mean_10');
      expect(bySplit[1].feature_name).toBe('sensor_15_std_20');
    });

    it('filters features by search query', () => {
      const searchRes = getFilteredFeatures(sampleFeatures, 'gain', 'all', 'sensor_4');
      expect(searchRes).toHaveLength(1);
      expect(searchRes[0].feature_name).toBe('sensor_4_mean_10');

      const statSearch = getFilteredFeatures(sampleFeatures, 'gain', 'all', 'r2');
      expect(statSearch).toHaveLength(1);
      expect(statSearch[0].feature_name).toBe('sensor_11_r2_75');
    });

    it('filters features by Top-N slice', () => {
      const top1 = getFilteredFeatures(sampleFeatures, 'gain', 1);
      expect(top1).toHaveLength(1);
      expect(top1[0].feature_name).toBe('sensor_11_r2_75');
    });

    it('correctly maps correlation quadrants and excludes un-evaluated features', () => {
      const comp = getCorrelationComparisonList(sampleFeatures, 'pearson');
      // sensor_15_std_20 has no correlation, so candidate count is 2
      expect(comp).toHaveLength(2);
      expect(comp[0].quadrant).toBe('high_importance_high_corr');
      expect(comp[0].quadrantLabel).toBe('High Gain / High Correlation');
      expect(comp[1].quadrant).toBe('high_importance_high_corr');
      expect(comp[1].quadrantLabel).toBe('High Gain / High Correlation');
    });

    it('pivots sensor trajectories for Recharts multi-line chart', () => {
      const trajs: SensorTrajectoryRecord[] = [
        { dataset: 'FD001', unit_id: 1, cycle: 1, sensor: 'sensor_11', sensor_value: 47.5 },
        { dataset: 'FD001', unit_id: 2, cycle: 1, sensor: 'sensor_11', sensor_value: 47.6 },
        { dataset: 'FD001', unit_id: 1, cycle: 2, sensor: 'sensor_11', sensor_value: 47.8 },
        { dataset: 'FD001', unit_id: 2, cycle: 2, sensor: 'sensor_11', sensor_value: 47.9 },
      ];

      const pivoted = getTrajectoryChartData(trajs, 'sensor_11', [1, 2]);
      expect(pivoted).toHaveLength(2);
      expect(pivoted[0].cycle).toBe(1);
      expect(pivoted[0]['unit_1']).toBe(47.5);
      expect(pivoted[0]['unit_2']).toBe(47.6);
      expect(pivoted[1].cycle).toBe(2);
      expect(pivoted[1]['unit_1']).toBe(47.8);
      expect(pivoted[1]['unit_2']).toBe(47.9);
    });

    it('computes engine sensor statistics accurately with net change semantics', () => {
      const trajs: SensorTrajectoryRecord[] = [
        { dataset: 'FD001', unit_id: 1, cycle: 1, sensor: 'sensor_11', sensor_value: 40.0 },
        { dataset: 'FD001', unit_id: 1, cycle: 2, sensor: 'sensor_11', sensor_value: 50.0 },
        { dataset: 'FD001', unit_id: 1, cycle: 3, sensor: 'sensor_11', sensor_value: 60.0 },
      ];

      const stats = getEngineSensorStatistics(trajs, 'sensor_11', 1);
      expect(stats).not.toBeNull();
      expect(stats?.count).toBe(3);
      expect(stats?.min).toBe(40.0);
      expect(stats?.max).toBe(60.0);
      expect(stats?.mean).toBe(50.0);
      expect(stats?.startCycle).toBe(1);
      expect(stats?.endCycle).toBe(3);
      expect(stats?.startValue).toBe(40.0);
      expect(stats?.endValue).toBe(60.0);
      expect(stats?.netChange).toBe(20.0); // 60.0 - 40.0
    });

    it('resolves sensor physical metadata and fallback gracefully', () => {
      const spec4 = getSensorSpec('sensor_4');
      expect(spec4.name).toBe('Total temp at LPT outlet');
      expect(spec4.unit).toBe('°R');

      const spec21 = getSensorSpec('sensor_21');
      expect(spec21.name).toBe('LPT coolant bleed');
      expect(spec21.unit).toBe('lbm/s');

      const spec11 = getSensorSpec('sensor_11');
      expect(spec11.name).toContain('Static pressure');
      expect(spec11.unit).toBe('psia');

      const fallbackSpec = getSensorSpec('sensor_99');
      expect(fallbackSpec.name).toBe('sensor_99');
    });
  });

  describe('6. Correlation Metric Comparison & Missing Value Regression Audit', () => {
    const mockAuditFeatures: FeatureImportanceRecord[] = [
      {
        dataset: 'FD001',
        feature_name: 'feat_both',
        sensor: 'sensor_11',
        statistic: 'mean',
        window_or_lag: '10',
        description: 'both correlations present',
        gain_importance: 1000,
        split_importance: 50,
        gain_importance_pct: 5.0,
        split_importance_pct: 5.0,
        gain_rank: 1,
        split_rank: 1,
        correlation_pearson: 0.65,
        correlation_spearman: 0.70,
        abs_pearson: 0.65,
        abs_spearman: 0.70,
      },
      {
        dataset: 'FD001',
        feature_name: 'feat_pearson_only',
        sensor: 'sensor_4',
        statistic: 'mean',
        window_or_lag: '10',
        description: 'pearson only',
        gain_importance: 500,
        split_importance: 20,
        gain_importance_pct: 2.5,
        split_importance_pct: 2.0,
        gain_rank: 2,
        split_rank: 2,
        correlation_pearson: -0.55,
        correlation_spearman: null,
        abs_pearson: 0.55,
        abs_spearman: null,
      },
      {
        dataset: 'FD001',
        feature_name: 'feat_spearman_only',
        sensor: 'sensor_8',
        statistic: 'max',
        window_or_lag: '20',
        description: 'spearman only',
        gain_importance: 300,
        split_importance: 10,
        gain_importance_pct: 1.5,
        split_importance_pct: 1.0,
        gain_rank: 3,
        split_rank: 3,
        correlation_pearson: null,
        correlation_spearman: -0.60,
        abs_pearson: null,
        abs_spearman: 0.60,
      },
      {
        dataset: 'FD001',
        feature_name: 'feat_neither',
        sensor: 'sensor_15',
        statistic: 'std',
        window_or_lag: '30',
        description: 'neither correlation available',
        gain_importance: 100,
        split_importance: 5,
        gain_importance_pct: 0.5,
        split_importance_pct: 0.5,
        gain_rank: 4,
        split_rank: 4,
        correlation_pearson: null,
        correlation_spearman: null,
        abs_pearson: null,
        abs_spearman: null,
      },
    ];

    it('Pearson view includes only records with valid Pearson and excludes missing Pearson without fallback', () => {
      const pearsonItems = getCorrelationComparisonList(mockAuditFeatures, 'pearson');
      const names = pearsonItems.map((r) => r.feature_name);

      expect(names).toContain('feat_both');
      expect(names).toContain('feat_pearson_only');
      expect(names).not.toContain('feat_spearman_only');
      expect(names).not.toContain('feat_neither');
      expect(pearsonItems).toHaveLength(2);

      // Verify that values are strictly Pearson
      const itemPearsonOnly = pearsonItems.find((r) => r.feature_name === 'feat_pearson_only');
      expect(itemPearsonOnly?.abs_pearson).toBe(0.55);
      expect(itemPearsonOnly?.abs_spearman).toBeNull();
    });

    it('Spearman view includes only records with valid Spearman and excludes missing Spearman without fallback', () => {
      const spearmanItems = getCorrelationComparisonList(mockAuditFeatures, 'spearman');
      const names = spearmanItems.map((r) => r.feature_name);

      expect(names).toContain('feat_both');
      expect(names).toContain('feat_spearman_only');
      expect(names).not.toContain('feat_pearson_only');
      expect(names).not.toContain('feat_neither');
      expect(spearmanItems).toHaveLength(2);

      // Verify that values are strictly Spearman
      const itemSpearmanOnly = spearmanItems.find((r) => r.feature_name === 'feat_spearman_only');
      expect(itemSpearmanOnly?.abs_spearman).toBe(0.60);
      expect(itemSpearmanOnly?.abs_pearson).toBeNull();
    });

    it('missing correlations never map to 0 on the scatter axis', () => {
      const pearsonItems = getCorrelationComparisonList(mockAuditFeatures, 'pearson');
      for (const item of pearsonItems) {
        expect(item.abs_pearson).not.toBeNull();
        expect(item.abs_pearson).toBeGreaterThan(0);
      }

      const spearmanItems = getCorrelationComparisonList(mockAuditFeatures, 'spearman');
      for (const item of spearmanItems) {
        expect(item.abs_spearman).not.toBeNull();
        expect(item.abs_spearman).toBeGreaterThan(0);
      }
    });

    it('verifies FD001 exact correlation availability counts from real artifact', () => {
      const artifactPath = path.resolve(__dirname, '../../reports/dashboard_stage4/feature_importance_FD001.csv');
      if (fs.existsSync(artifactPath)) {
        const rawCsv = fs.readFileSync(artifactPath, 'utf-8');
        const fd001Features = parseFeatureImportanceCsv(rawCsv, 'FD001');

        expect(fd001Features).toHaveLength(670);

        let bothCount = 0;
        let pearsonOnlyCount = 0;
        let spearmanOnlyCount = 0;
        let neitherCount = 0;

        for (const f of fd001Features) {
          const hasP = f.abs_pearson !== null;
          const hasS = f.abs_spearman !== null;
          if (hasP && hasS) bothCount++;
          else if (hasP && !hasS) pearsonOnlyCount++;
          else if (!hasP && hasS) spearmanOnlyCount++;
          else neitherCount++;
        }

        // Exact regression test expectations mandated by audit
        expect(bothCount).toBe(39);
        expect(pearsonOnlyCount).toBe(11);
        expect(spearmanOnlyCount).toBe(11);
        expect(neitherCount).toBe(609);
        expect(bothCount + pearsonOnlyCount + spearmanOnlyCount + neitherCount).toBe(670);

        // Visible feature counts per metric
        const evaluatedPearson = getCorrelationComparisonList(fd001Features, 'pearson');
        expect(evaluatedPearson).toHaveLength(50); // 39 + 11

        const evaluatedSpearman = getCorrelationComparisonList(fd001Features, 'spearman');
        expect(evaluatedSpearman).toHaveLength(50); // 39 + 11
      }
    });
  });

  describe('7. Raw Sensor Trajectories Presentation & Sampling Fidelity', () => {
    it('verifies initial cycle 1 and terminal cycle retention with deterministic stride 2', () => {
      const trajPath = path.resolve(__dirname, '../../reports/dashboard_stage4/sensor_trajectories_FD001.csv');
      const manifestPath = path.resolve(__dirname, '../../reports/dashboard_stage4/sensor_trajectories_manifest.json');

      if (fs.existsSync(trajPath) && fs.existsSync(manifestPath)) {
        const rawCsv = fs.readFileSync(trajPath, 'utf-8');
        const trajs = parseSensorTrajectoriesCsv(rawCsv, 'FD001');
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

        const selectedUnits: number[] = manifest.subsets.FD001.selected_units;
        expect(selectedUnits).toHaveLength(10);

        for (const u of selectedUnits) {
          const unitTrajs = trajs.filter((r) => r.unit_id === u && r.sensor === 'sensor_11');
          expect(unitTrajs.length).toBeGreaterThan(0);

          // Cycle 1 must be present
          const cycles = unitTrajs.map((r) => r.cycle);
          expect(cycles[0]).toBe(1);

          // Terminal failure cycle must match manifest lifetime
          const expectedLifetime = manifest.subsets.FD001.unit_metadata[String(u)].lifetime_cycles;
          expect(cycles[cycles.length - 1]).toBe(expectedLifetime);

          // All intermediate cycles follow stride 2 progression
          for (let i = 1; i < cycles.length - 1; i++) {
            expect(cycles[i] % 2).toBe(1); // odd cycles 1, 3, 5...
          }
        }
      }
    });
  });
});
