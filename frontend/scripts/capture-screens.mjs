import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const outputDir = path.resolve(__dirname, '..', 'screenshots');

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

function resolveChromePath() {
  if (process.env.CHROME_EXECUTABLE) {
    if (fs.existsSync(process.env.CHROME_EXECUTABLE)) {
      return process.env.CHROME_EXECUTABLE;
    }
    throw new Error(
      `Specified CHROME_EXECUTABLE does not exist at: ${process.env.CHROME_EXECUTABLE}`
    );
  }

  const candidatePaths = [
    // Windows common locations
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    process.env.LOCALAPPDATA
      ? path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe')
      : '',
    // macOS common locations
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    // Linux common locations
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);

  for (const candidate of candidatePaths) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  throw new Error(
    'Chrome executable could not be located automatically.\n' +
    'Please set the CHROME_EXECUTABLE environment variable to point to your Chrome or Chromium binary.'
  );
}

async function waitForServer(url, timeout = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  throw new Error(`Server did not respond within ${timeout}ms at ${url}`);
}

async function runCapture() {
  const chromePath = resolveChromePath();
  console.log(`[capture-screens] Using Chrome binary at: ${chromePath}`);

  console.log('[capture-screens] Starting Vite preview server...');
  const viteBin = path.resolve(__dirname, '..', 'node_modules', 'vite', 'bin', 'vite.js');
  const server = spawn(process.execPath, [viteBin, 'preview', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], {
    cwd: path.resolve(__dirname, '..'),
    env: { ...process.env, NODE_ENV: 'production' },
  });

  server.stdout.on('data', (d) => console.log(`[vite] ${d}`));
  server.stderr.on('data', (d) => console.error(`[vite error] ${d}`));

  try {
    await waitForServer('http://127.0.0.1:4173');
    console.log('[capture-screens] Server is ready. Launching headless Chrome...');

    const browser = await puppeteer.launch({
      executablePath: chromePath,
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--disable-crash-reporter',
        '--no-first-run',
        '--no-default-browser-check'
      ]
    });

    const page = await browser.newPage();

    // 1. Overview Screen Viewports (1440, 1280, 768, 390)
    console.log('[capture-screens] Capturing Overview (desktop-1440)...');
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto('http://127.0.0.1:4173/?view=overview&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });
    await page.screenshot({ path: path.join(outputDir, 'desktop-1440.png'), fullPage: false });

    console.log('[capture-screens] Capturing Overview (desktop-1280)...');
    await page.setViewport({ width: 1280, height: 800 });
    await page.goto('http://127.0.0.1:4173/?view=overview&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });
    await page.screenshot({ path: path.join(outputDir, 'desktop-1280.png'), fullPage: false });

    console.log('[capture-screens] Capturing Overview (tablet-768)...');
    await page.setViewport({ width: 768, height: 1024 });
    await page.goto('http://127.0.0.1:4173/?view=overview&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });
    await page.screenshot({ path: path.join(outputDir, 'tablet-768.png'), fullPage: false });

    console.log('[capture-screens] Capturing Overview (mobile-390)...');
    await page.setViewport({ width: 390, height: 844 });
    await page.goto('http://127.0.0.1:4173/?view=overview&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });
    await page.screenshot({ path: path.join(outputDir, 'mobile-390.png'), fullPage: false });

    // 2. Engine Predictions Workspace Captures
    console.log('[capture-screens] Capturing Predictions Workspace (predictions-desktop-1440)...');
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto('http://127.0.0.1:4173/?view=predictions&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });
    await page.screenshot({ path: path.join(outputDir, 'predictions-desktop-1440.png'), fullPage: false });

    console.log('[capture-screens] Capturing Predictions with Engine Inspected (predictions-desktop-1440-selected)...');
    const outlierBtn = await page.$('button[aria-label^="Inspect outlier unit"]');
    if (outlierBtn) {
      await outlierBtn.click();
    } else {
      const inspectBtn = await page.$('.data-table tbody tr button');
      if (inspectBtn) await inspectBtn.click();
    }
    await new Promise((r) => setTimeout(r, 400));
    const inspector = await page.$('[aria-label^="Engine endpoint inspection"]');
    if (inspector) {
      await inspector.evaluate((el) => el.scrollIntoView({ behavior: 'instant', block: 'center' }));
    }
    await page.screenshot({ path: path.join(outputDir, 'predictions-desktop-1440-selected.png'), fullPage: false });

    // 3. Mobile Predictions & Dataset Switching Verification (390x844)
    console.log('[capture-screens] Capturing Predictions Mobile (predictions-mobile-390)...');
    await page.setViewport({ width: 390, height: 844 });
    await page.goto('http://127.0.0.1:4173/?view=predictions&dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table', { timeout: 5000 });

    // Check no page-level horizontal overflow
    const hasHorizontalOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    console.log(`[capture-screens] Mobile 390px horizontal overflow check: ${hasHorizontalOverflow ? 'FAILED (overflow)' : 'PASSED (no overflow)'}`);
    await page.screenshot({ path: path.join(outputDir, 'predictions-mobile-390.png'), fullPage: false });

    console.log('[capture-screens] Capturing Predictions Mobile Inspector (predictions-mobile-390-inspector)...');
    const mobileOutlierBtn = await page.$('button[aria-label^="Inspect outlier unit"]');
    if (mobileOutlierBtn) {
      await mobileOutlierBtn.click();
    } else {
      const mobileInspectBtn = await page.$('.data-table tbody tr button');
      if (mobileInspectBtn) await mobileInspectBtn.click();
    }
    await new Promise((r) => setTimeout(r, 400));
    const mobileInspector = await page.$('[aria-label^="Engine endpoint inspection"]');
    if (mobileInspector) {
      await mobileInspector.evaluate((el) => el.scrollIntoView({ behavior: 'instant', block: 'center' }));
    }
    await page.screenshot({ path: path.join(outputDir, 'predictions-mobile-390-inspector.png'), fullPage: false });

    // Verify all 4 datasets are selectable on mobile
    console.log('[capture-screens] Testing all 4 datasets on mobile (FD001 -> FD002 -> FD003 -> FD004)...');
    const datasets = ['FD002', 'FD003', 'FD004', 'FD001'];
    for (const ds of datasets) {
      // Find mobile native select or button
      const mobileSelect = await page.$('.dataset-native-select');
      if (mobileSelect) {
        await page.select('.dataset-native-select', ds);
      } else {
        await page.click(`button[aria-pressed]:has-text("${ds}")`);
      }
      await new Promise((r) => setTimeout(r, 300));
      const currentUrl = page.url();
      console.log(`[capture-screens] Selected ${ds} on mobile, URL: ${currentUrl}`);
    }

    // Capture mobile FD004 to verify FD004 accessibility
    await page.select('.dataset-native-select', 'FD004');
    await new Promise((r) => setTimeout(r, 400));
    await page.screenshot({ path: path.join(outputDir, 'predictions-mobile-390-fd004.png'), fullPage: false });

    await browser.close();
    console.log('[capture-screens] All Stage 2 verification screenshots captured successfully!');
  } finally {
    server.kill();
    console.log('[capture-screens] Server stopped.');
  }
}

runCapture().catch((err) => {
  console.error('[capture-screens] Error:', err);
  process.exit(1);
});
