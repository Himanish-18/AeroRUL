/**
 * AeroRUL — Research Artifact Synchronization Script
 *
 * Copies verified Phase 4 and Phase 5 benchmark outputs from reports/ into
 * frontend/public/data/ for runtime consumption by the dashboard and evaluation workbench.
 *
 * Strict requirements:
 * 1. Resolved relative to this script file (never CWD).
 * 2. Strict allowlist only — no binaries or extraneous files.
 * 3. Atomic copy with existence and size verification.
 * 4. Fails with non-zero exit code if any artifact is missing or empty.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Project root is two levels up from frontend/scripts/
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');

const SYNC_TARGETS = [
  {
    phase: 'phase4',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'phase4'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'phase4'),
    allowlist: [
      'model_benchmark_metrics.csv',
    ],
  },
  {
    phase: 'phase5',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'phase5'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'phase5'),
    allowlist: [
      'final_test_metrics.csv',
      'final_test_predictions.csv',
      'final_model_manifest.json',
      'cross_validation_metrics.csv',
      'development_validation_metrics.csv',
      'hyperparameter_search_results.csv',
      'window_ablation_metrics.csv',
      'target_cap_sensitivity.csv',
    ],
  },
];

console.log('[sync-artifacts] Starting research artifact synchronization...');

let totalSynced = 0;
let totalExpected = 0;

for (const target of SYNC_TARGETS) {
  console.log(`[sync-artifacts] Processing ${target.phase}...`);
  console.log(`  Source: ${target.sourceDir}`);
  console.log(`  Target: ${target.destDir}`);

  if (!fs.existsSync(target.sourceDir)) {
    console.error(`[sync-artifacts] ERROR: Source directory not found: ${target.sourceDir}`);
    process.exit(1);
  }

  if (!fs.existsSync(target.destDir)) {
    fs.mkdirSync(target.destDir, { recursive: true });
    console.log(`  Created directory: ${target.destDir}`);
  }

  for (const filename of target.allowlist) {
    totalExpected++;
    const sourcePath = path.join(target.sourceDir, filename);
    const destPath = path.join(target.destDir, filename);

    if (!fs.existsSync(sourcePath)) {
      console.error(`[sync-artifacts] ERROR: Required artifact missing: ${sourcePath}`);
      process.exit(1);
    }

    const stat = fs.statSync(sourcePath);
    if (stat.size === 0) {
      console.error(`[sync-artifacts] ERROR: Artifact is empty (0 bytes): ${sourcePath}`);
      process.exit(1);
    }

    try {
      fs.copyFileSync(sourcePath, destPath);
      const destStat = fs.statSync(destPath);
      if (destStat.size !== stat.size) {
        throw new Error(`Size mismatch after copy (${destStat.size} vs ${stat.size} bytes)`);
      }
      console.log(`  Synced: ${filename} (${(stat.size / 1024).toFixed(1)} KB)`);
      totalSynced++;
    } catch (err) {
      console.error(`[sync-artifacts] ERROR: Failed to copy ${filename}:`, err);
      process.exit(1);
    }
  }
}

console.log(`[sync-artifacts] Successfully synchronized ${totalSynced}/${totalExpected} artifacts across all targets.`);
