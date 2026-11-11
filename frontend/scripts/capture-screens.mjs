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

async function waitForServer(url, timeout = 10000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // wait 200ms
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  throw new Error(`Server did not respond within ${timeout}ms at ${url}`);
}

async function runCapture() {
  console.log('[capture-screens] Starting Vite preview server...');
  const viteBin = path.resolve(__dirname, '..', 'node_modules', 'vite', 'bin', 'vite.js');
  const server = spawn(process.execPath, [viteBin, 'preview', '--port', '4173', '--strictPort'], {
    cwd: path.resolve(__dirname, '..'),
    env: { ...process.env, NODE_ENV: 'production' },
  });

  server.stdout.on('data', (d) => console.log(`[vite] ${d}`));
  server.stderr.on('data', (d) => console.error(`[vite error] ${d}`));

  try {
    await waitForServer('http://localhost:4173');
    console.log('[capture-screens] Server is ready. Launching headless Chrome...');

    const browser = await puppeteer.launch({
      executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
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

    const viewports = [
      { name: 'desktop-1440', width: 1440, height: 900 },
      { name: 'desktop-1280', width: 1280, height: 800 },
      { name: 'tablet-768', width: 768, height: 1024 },
      { name: 'mobile-390', width: 390, height: 844 }
    ];

    for (const vp of viewports) {
      console.log(`[capture-screens] Capturing ${vp.name} (${vp.width}x${vp.height})...`);
      await page.setViewport({ width: vp.width, height: vp.height });
      await page.goto('http://localhost:4173/?dataset=FD001', { waitUntil: 'networkidle0' });

      // Wait for table and chart elements to ensure populated state
      await page.waitForSelector('.data-table', { timeout: 5000 });

      const filepath = path.join(outputDir, `${vp.name}.png`);
      await page.screenshot({ path: filepath, fullPage: false });
      console.log(`[capture-screens] Saved: ${filepath}`);
    }

    // Capture interaction: Click first table row to inspect engine
    console.log('[capture-screens] Testing engine selection interaction...');
    await page.setViewport({ width: 1440, height: 900 });
    await page.goto('http://localhost:4173/?dataset=FD001', { waitUntil: 'networkidle0' });
    await page.waitForSelector('.data-table tbody tr');
    await page.click('.data-table tbody tr:first-child');
    await new Promise((r) => setTimeout(r, 400)); // wait for inspector render
    const selectedFilepath = path.join(outputDir, 'desktop-1440-selected-engine.png');
    await page.screenshot({ path: selectedFilepath, fullPage: false });
    console.log(`[capture-screens] Saved selected engine capture: ${selectedFilepath}`);

    await browser.close();
    console.log('[capture-screens] All screenshots captured successfully!');
  } finally {
    server.kill();
    console.log('[capture-screens] Server stopped.');
  }
}

runCapture().catch((err) => {
  console.error('[capture-screens] Error:', err);
  process.exit(1);
});
