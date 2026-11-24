/**
 * AeroRUL — Dataset Explorer Parser & Adapter Test Suite
 */

import { describe, expect, it, beforeEach, vi } from 'vitest';
import {
  parseEngineIndex,
  parseRawObservationText,
  parseRulVectorText,
  validateObservationRecords,
} from '../src/lib/data/datasetExplorerParser';
import {
  filterAndSortEngines,
  filterObservations,
  derivePhysicalTrainingRulSeries,
  computeNumericDistribution,
  extractTestEndpointRuls,
  computeTestEndpointRulDistribution,
} from '../src/lib/data/datasetExplorerAdapter';
import {
  loadRawObservations,
  clearDatasetExplorerCache,
} from '../src/lib/data/datasetExplorerLoader';
import {
  EngineIndexRecord,
} from '../src/lib/data/datasetExplorerTypes';

// Mock data generator for 26-column rows
function generateMockRawRow(unitId: number, cycle: number, s1 = 518.67, s2 = 642.15): string {
  const op1 = 0.0001;
  const op2 = 0.0002;
  const op3 = 100.0;
  const sensors = [
    s1, s2, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3,
    47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100, 39.06, 23.419,
  ];
  return `${unitId} ${cycle} ${op1} ${op2} ${op3} ${sensors.join(' ')}`;
}

describe('Dataset Explorer — Strict Parser Tests', () => {
  it('parses valid 26-column whitespace-separated raw records into numbers', () => {
    const rawText = [
      generateMockRawRow(1, 1),
      generateMockRawRow(1, 2),
      generateMockRawRow(2, 1),
    ].join('\n');

    const records = parseRawObservationText(rawText);
    expect(records).toHaveLength(3);
    expect(records[0].unit_id).toBe(1);
    expect(records[0].cycle).toBe(1);
    expect(records[0].op_setting_3).toBe(100.0);
    expect(records[0].sensor_1).toBeCloseTo(518.67);
    expect(records[1].cycle).toBe(2);
    expect(records[2].unit_id).toBe(2);
  });

  it('rejects lines with fewer than 26 columns and includes line number', () => {
    const textWithMalformed = [
      generateMockRawRow(1, 1),
      '1 2 0.001 0.002', // only 4 columns on line 2
      generateMockRawRow(1, 3),
    ].join('\n');

    expect(() => parseRawObservationText(textWithMalformed)).toThrowError(
      /Parse error on line 2: expected exactly 26 columns, got 4/
    );
  });

  it('rejects lines with more than 26 columns without truncating', () => {
    const textWithExtraCol = [
      generateMockRawRow(1, 1),
      `${generateMockRawRow(1, 2)} 999.99`, // 27 columns on line 2
    ].join('\n');

    expect(() => parseRawObservationText(textWithExtraCol)).toThrowError(
      /Parse error on line 2: expected exactly 26 columns, got 27/
    );
  });

  it('rejects invalid non-positive or corrupted unit ID', () => {
    const badUnitId1 = generateMockRawRow(1, 1).replace(/^1 /, '0 ');
    expect(() => parseRawObservationText(badUnitId1)).toThrowError(
      /Parse error on line 1: unit_id must be a positive integer/
    );

    const badUnitId2 = generateMockRawRow(1, 1).replace(/^1 /, '-5 ');
    expect(() => parseRawObservationText(badUnitId2)).toThrowError(
      /Parse error on line 1: unit_id must be a positive integer/
    );

    const badUnitId3 = generateMockRawRow(1, 1).replace(/^1 /, '1.5 ');
    expect(() => parseRawObservationText(badUnitId3)).toThrowError(
      /Parse error on line 1: unit_id must be a positive integer/
    );

    const badUnitId4 = generateMockRawRow(1, 1).replace(/^1 /, '1invalid ');
    expect(() => parseRawObservationText(badUnitId4)).toThrowError(
      /Parse error on line 1: unit_id must be a positive integer/
    );
  });

  it('rejects invalid or corrupted cycle value', () => {
    const badCycle = generateMockRawRow(1, 1).replace(/^1 1 /, '1 0 ');
    expect(() => parseRawObservationText(badCycle)).toThrowError(
      /Parse error on line 1: cycle must be a positive integer/
    );

    const badCycle2 = generateMockRawRow(1, 1).replace(/^1 1 /, '1 cycleX ');
    expect(() => parseRawObservationText(badCycle2)).toThrowError(
      /Parse error on line 1: cycle must be a positive integer/
    );
  });

  it('rejects numeric-prefix corruption in floating-point values', () => {
    const corruptedFloat = generateMockRawRow(1, 1).replace('518.67', '518.67invalid');
    expect(() => parseRawObservationText(corruptedFloat)).toThrowError(
      /Parse error on line 1: sensor_1 must be a valid finite number/
    );
  });

  it('rejects NaN, Infinity, and -Infinity values', () => {
    const nanRow = generateMockRawRow(1, 1).replace('518.67', 'NaN');
    expect(() => parseRawObservationText(nanRow)).toThrowError(
      /Parse error on line 1: sensor_1 must be a valid finite number/
    );

    const infRow = generateMockRawRow(1, 1).replace('518.67', 'Infinity');
    expect(() => parseRawObservationText(infRow)).toThrowError(
      /Parse error on line 1: sensor_1 must be a valid finite number/
    );

    const negInfRow = generateMockRawRow(1, 1).replace('518.67', '-Infinity');
    expect(() => parseRawObservationText(negInfRow)).toThrowError(
      /Parse error on line 1: sensor_1 must be a valid finite number/
    );
  });

  it('skips empty lines while accurately preserving subsequent line numbers', () => {
    const textWithEmptyLines = [
      '',
      generateMockRawRow(1, 1), // line 2
      '',
      '',
      'bad line with 3 cols',   // line 5 (5 words)
    ].join('\n');

    expect(() => parseRawObservationText(textWithEmptyLines)).toThrowError(
      /Parse error on line 5: expected exactly 26 columns, got 5/
    );
  });

  it('handles empty or blank text gracefully', () => {
    expect(parseRawObservationText('')).toEqual([]);
    expect(parseRawObservationText('   \n\n  ')).toEqual([]);
  });

  it('parses RUL vector text strictly and rejects non-integer strings', () => {
    const rulText = '112\n98\n69\n82\n';
    const vec = parseRulVectorText(rulText);
    expect(vec).toEqual([112, 98, 69, 82]);

    expect(() => parseRulVectorText('112\n98invalid\n69')).toThrowError(
      /Parse error on line 2: RUL value must be a non-negative integer/
    );
  });

  it('parses engine index CSV text correctly preserving missing/null semantics', () => {
    const csv = [
      'dataset,split,unit_id,observation_count,first_cycle,last_observed_cycle,observed_cycle_count,endpoint_status,physical_failure_cycle,official_endpoint_rul',
      'FD001,train,1,192,1,192,192,run-to-failure,192,',
      'FD001,test,1,31,1,31,31,censored,,112',
    ].join('\n');

    const records = parseEngineIndex(csv);
    expect(records).toHaveLength(2);

    // Train record
    expect(records[0].dataset).toBe('FD001');
    expect(records[0].split).toBe('train');
    expect(records[0].unit_id).toBe(1);
    expect(records[0].endpoint_status).toBe('run-to-failure');
    expect(records[0].physical_failure_cycle).toBe(192);
    expect(records[0].official_endpoint_rul).toBeNull();

    // Test record
    expect(records[1].split).toBe('test');
    expect(records[1].endpoint_status).toBe('censored');
    expect(records[1].physical_failure_cycle).toBeNull();
    expect(records[1].official_endpoint_rul).toBe(112);
  });

  it('validates observation records structural integrity and monotonic cycles', () => {
    const goodRecords = [
      { ...parseRawObservationText(generateMockRawRow(1, 1))[0] },
      { ...parseRawObservationText(generateMockRawRow(1, 2))[0] },
      { ...parseRawObservationText(generateMockRawRow(2, 1))[0] },
    ];

    const quality = validateObservationRecords(goodRecords);
    expect(quality.column_count_ok).toBe(true);
    expect(quality.total_missing_values).toBe(0);
    expect(quality.total_non_finite_values).toBe(0);
    expect(quality.cycles_ordered).toBe(true);
    expect(quality.cycles_start_at_1).toBe(true);
    expect(quality.valid_unit_ids).toBe(true);
    expect(quality.valid_cycle_values).toBe(true);
    expect(quality.structural_status).toBe('PASS');
    expect(quality.integrity_status).toBe('PASS');
  });
});

describe('Dataset Explorer — Task 1: Test Endpoint RUL Histogram & Label Alignment Tests', () => {
  it('extracts exactly 100 endpoint labels for FD001', () => {
    const ruls = Array.from({ length: 100 }, (_, i) => 10 + (i % 80));
    const extracted = extractTestEndpointRuls(ruls);
    expect(extracted).toHaveLength(100);
    expect(extracted[0].unitId).toBe(1);
    expect(extracted[99].unitId).toBe(100);
  });

  it('extracts exactly 259 endpoint labels for FD002', () => {
    const ruls = Array.from({ length: 259 }, (_, i) => 15 + (i % 100));
    const extracted = extractTestEndpointRuls(ruls);
    expect(extracted).toHaveLength(259);
    expect(extracted[0].unitId).toBe(1);
    expect(extracted[258].unitId).toBe(259);
  });

  it('extracts exactly 100 endpoint labels for FD003', () => {
    const ruls = Array.from({ length: 100 }, (_, i) => 20 + (i % 70));
    const extracted = extractTestEndpointRuls(ruls);
    expect(extracted).toHaveLength(100);
    expect(extracted[0].unitId).toBe(1);
    expect(extracted[99].unitId).toBe(100);
  });

  it('extracts exactly 248 endpoint labels for FD004', () => {
    const ruls = Array.from({ length: 248 }, (_, i) => 12 + (i % 90));
    const extracted = extractTestEndpointRuls(ruls);
    expect(extracted).toHaveLength(248);
    expect(extracted[0].unitId).toBe(1);
    expect(extracted[247].unitId).toBe(248);
  });

  it('computes test endpoint RUL distribution where histogram total count equals test engine count', () => {
    const testRecords: EngineIndexRecord[] = Array.from({ length: 100 }, (_, i) => ({
      dataset: 'FD001',
      split: 'test',
      unit_id: i + 1,
      observation_count: 50 + (i % 30),
      first_cycle: 1,
      last_observed_cycle: 50 + (i % 30),
      observed_cycle_count: 50 + (i % 30),
      endpoint_status: 'censored',
      physical_failure_cycle: null,
      official_endpoint_rul: 20 + (i % 100),
    }));

    const bins = computeTestEndpointRulDistribution(testRecords, 16);
    expect(bins.length).toBeGreaterThan(0);
    const totalCount = bins.reduce((acc, b) => acc + b.count, 0);
    expect(totalCount).toBe(100);
  });

  it('ensures no test per-cycle RUL records are generated', () => {
    const testRecords: EngineIndexRecord[] = [
      {
        dataset: 'FD001',
        split: 'test',
        unit_id: 1,
        observation_count: 31,
        first_cycle: 1,
        last_observed_cycle: 31,
        observed_cycle_count: 31,
        endpoint_status: 'censored',
        physical_failure_cycle: null,
        official_endpoint_rul: 112,
      },
    ];

    const extracted = extractTestEndpointRuls(testRecords);
    expect(extracted).toHaveLength(1); // exactly 1 endpoint record, not 31 per-cycle records
    expect(extracted[0]).toEqual({ unitId: 1, endpointRul: 112 });
  });
});

describe('Dataset Explorer — Loader Manifest Verification Tests', () => {
  beforeEach(() => {
    clearDatasetExplorerCache();
  });

  it('throws and does not cache when parsed record count mismatches expected row count', async () => {
    // Generate 3 rows while expecting 5
    const rawContent = [
      generateMockRawRow(1, 1),
      generateMockRawRow(1, 2),
      generateMockRawRow(1, 3),
    ].join('\n');

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('train_FD001.txt.gz')) {
        return Promise.resolve(new Response(rawContent, { status: 200 }));
      }
      return Promise.resolve(new Response('Not found', { status: 404 }));
    });

    await expect(
      loadRawObservations('FD001', 'train', '/data/datasets', 5)
    ).rejects.toThrowError(
      /Dataset integrity violation: parsed 3 records for FD001 \(train\), but manifest expects 5 rows/
    );
  });
});

describe('Dataset Explorer — Adapter & Filtering Tests', () => {
  const mockEngines: EngineIndexRecord[] = [
    {
      dataset: 'FD001',
      split: 'train',
      unit_id: 1,
      observation_count: 192,
      first_cycle: 1,
      last_observed_cycle: 192,
      observed_cycle_count: 192,
      endpoint_status: 'run-to-failure',
      physical_failure_cycle: 192,
      official_endpoint_rul: null,
    },
    {
      dataset: 'FD001',
      split: 'train',
      unit_id: 2,
      observation_count: 287,
      first_cycle: 1,
      last_observed_cycle: 287,
      observed_cycle_count: 287,
      endpoint_status: 'run-to-failure',
      physical_failure_cycle: 287,
      official_endpoint_rul: null,
    },
    {
      dataset: 'FD001',
      split: 'train',
      unit_id: 10,
      observation_count: 166,
      first_cycle: 1,
      last_observed_cycle: 166,
      observed_cycle_count: 166,
      endpoint_status: 'run-to-failure',
      physical_failure_cycle: 166,
      official_endpoint_rul: null,
    },
  ];

  it('filters engines by search query matching unit id substring', () => {
    const res1 = filterAndSortEngines(mockEngines, {
      searchQuery: '10',
      sortBy: 'unit_id',
      sortAsc: true,
      page: 1,
      pageSize: 10,
    });
    expect(res1.items).toHaveLength(1);
    expect(res1.items[0].unit_id).toBe(10);

    const res2 = filterAndSortEngines(mockEngines, {
      searchQuery: 'Unit #1',
      sortBy: 'unit_id',
      sortAsc: true,
      page: 1,
      pageSize: 10,
    });
    // matches 1 and 10
    expect(res2.items.map((i) => i.unit_id)).toEqual([1, 10]);
  });

  it('sorts engines by observation count ascending and descending', () => {
    const asc = filterAndSortEngines(mockEngines, {
      searchQuery: '',
      sortBy: 'observation_count',
      sortAsc: true,
      page: 1,
      pageSize: 10,
    });
    expect(asc.items.map((i) => i.observation_count)).toEqual([166, 192, 287]);

    const desc = filterAndSortEngines(mockEngines, {
      searchQuery: '',
      sortBy: 'observation_count',
      sortAsc: false,
      page: 1,
      pageSize: 10,
    });
    expect(desc.items.map((i) => i.observation_count)).toEqual([287, 192, 166]);
  });

  it('paginates engine records with custom page sizes', () => {
    const p1 = filterAndSortEngines(mockEngines, {
      searchQuery: '',
      sortBy: 'unit_id',
      sortAsc: true,
      page: 1,
      pageSize: 2,
    });
    expect(p1.items).toHaveLength(2);
    expect(p1.totalPages).toBe(2);
    expect(p1.totalCount).toBe(3);

    const p2 = filterAndSortEngines(mockEngines, {
      searchQuery: '',
      sortBy: 'unit_id',
      sortAsc: true,
      page: 2,
      pageSize: 2,
    });
    expect(p2.items).toHaveLength(1);
    expect(p2.items[0].unit_id).toBe(10);
  });

  it('filters raw observations by selected engine and cycle range', () => {
    const rawRows = [
      parseRawObservationText(generateMockRawRow(1, 10))[0],
      parseRawObservationText(generateMockRawRow(1, 20))[0],
      parseRawObservationText(generateMockRawRow(1, 30))[0],
      parseRawObservationText(generateMockRawRow(2, 10))[0],
    ];

    const res = filterObservations(rawRows, {
      unitId: 1,
      minCycle: 15,
      maxCycle: 35,
      selectedSensors: ['sensor_1'],
      sortBy: 'cycle',
      sortAsc: true,
      page: 1,
      pageSize: 25,
    });

    expect(res.totalCount).toBe(2);
    expect(res.items.map((r) => r.cycle)).toEqual([20, 30]);
    expect(res.allEngineCycles).toBe(3);
    expect(res.minObservedCycle).toBe(10);
    expect(res.maxObservedCycle).toBe(30);
  });

  it('derives physical training RUL series correctly with terminal cycle = 0', () => {
    const engineObs = [
      parseRawObservationText(generateMockRawRow(1, 1))[0],
      parseRawObservationText(generateMockRawRow(1, 50))[0],
      parseRawObservationText(generateMockRawRow(1, 100))[0],
    ];

    const series = derivePhysicalTrainingRulSeries(engineObs);
    expect(series).toHaveLength(3);
    expect(series[0]).toEqual({ cycle: 1, physicalRul: 99 });
    expect(series[1]).toEqual({ cycle: 50, physicalRul: 50 });
    expect(series[2]).toEqual({ cycle: 100, physicalRul: 0 }); // terminal failure cycle RUL = 0
  });

  it('computes distribution histogram bins for numeric arrays', () => {
    const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    const bins = computeNumericDistribution(values, 5);
    expect(bins).toHaveLength(5);
    const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
    expect(totalBinned).toBe(values.length);
  });
});
