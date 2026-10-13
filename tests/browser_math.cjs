// End-to-end checks use the same MathJax script and Pages subpath as production.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const prefix = '/multimodal-atlas/';
const mime = {'.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.png': 'image/png', '.pdf': 'application/pdf'};

async function main() {
  const server = http.createServer(async (req, res) => {
    try {
      const pathname = new URL(req.url, 'http://localhost').pathname;
      if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
      const filename = path.resolve(root, pathname.slice(prefix.length) || 'index.html');
      if (!filename.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
      const data = await fs.readFile(filename);
      res.setHeader('Content-Type', mime[path.extname(filename)] || 'application/octet-stream');
      res.end(data);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, colorScheme: 'light'});
    const base = `http://127.0.0.1:${server.address().port}${prefix}`;
    const slugs = (await fs.readdir(path.join(root, 'reports'))).sort();
    const results = [];
    await fs.mkdir(path.join(root, 'test-results'), {recursive: true});
    for (const slug of slugs) {
      await page.goto(`${base}reports/${slug}/index.html`);
      await page.waitForFunction(() => window.MathJax?.startup?.promise, null, {timeout: 30000});
      await page.evaluate(() => MathJax.startup.promise);
      const stats = await page.evaluate(() => ({
        formulas: document.querySelectorAll('.arithmatex').length,
        rendered: document.querySelectorAll('.arithmatex mjx-container[jax="SVG"]').length,
        errors: [...document.querySelectorAll('[data-mjx-error], mjx-merror')].map(n => n.textContent),
      }));
      assert.equal(stats.rendered, stats.formulas, `${slug}: unrendered math`);
      assert.deepEqual(stats.errors, [], `${slug}: TeX errors`);
      for (const width of [1440, 390]) {
        await page.setViewportSize({width, height: 1000});
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        assert.ok(overflow <= 1, `${slug}: viewport ${width} overflowed by ${overflow}px`);
      }
      await page.setViewportSize({width: 1440, height: 1000});
      if (slug === '00-foundations' || slug === '01-vision-language') {
        const label = slug === '00-foundations' ? 'DPO（Direct Preference' : 'SigLIP（2023）';
        const paragraph = page.locator('article p').filter({hasText: label}).first();
        await paragraph.scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(root, 'test-results', `${slug}.png`)});
      }
      results.push({slug, ...stats});
      console.log(`${slug}: ${stats.rendered} formulas rendered`);
    }
    await page.locator('#theme-toggle').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    await fs.writeFile(path.join(root, 'test-results', 'browser-math.json'), JSON.stringify(results, null, 2));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
