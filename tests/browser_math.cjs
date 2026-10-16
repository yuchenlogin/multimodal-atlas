// End-to-end checks use the same MathJax script and Pages subpath as production.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const prefix = '/multimodal-atlas/';
const mime = {'.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.png': 'image/png', '.pdf': 'application/pdf'};

async function main() {
  const output = path.join(root, 'test-results', 'browser-math.json');
  await fs.mkdir(path.dirname(output), {recursive: true});
  const report = {status: 'running', startedAt: new Date().toISOString(), inputs: {}, results: []};
  await fs.writeFile(output, JSON.stringify(report, null, 2));
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
    const reports = (await fs.readdir(path.join(root, 'reports'))).sort();
    const inputs = ['index.html', 'paper/index.html', 'css/site.css', 'js/site.js', 'tests/browser_math.cjs',
      ...reports.flatMap(slug => [`reports/${slug}/index.html`, `reports/${slug}/report.md`]),
      'paper/proposal.md',
      ...(await fs.readdir(path.join(root, 'assets', 'technical'))).map(name => `assets/technical/${name}`)];
    for (const filename of inputs) {
      report.inputs[filename] = crypto.createHash('sha256').update(await fs.readFile(path.join(root, filename))).digest('hex');
    }
    await fs.writeFile(output, JSON.stringify(report, null, 2));
    browser = await chromium.launch({headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, colorScheme: 'light'});
    const base = `http://127.0.0.1:${server.address().port}${prefix}`;
    const slugs = (await fs.readdir(path.join(root, 'reports'))).sort();
    const pages = [];
    for (const slug of slugs) {
      const source = await fs.readFile(path.join(root, 'reports', slug, 'report.md'), 'utf8');
      pages.push({slug, url: `reports/${slug}/index.html`, figures: [...source.matchAll(/^!\[.*\]\([^\r\n)]+\)\s*$/gm)].length});
    }
    pages.push({slug: 'proposal', url: 'paper/index.html', figures: 0});
    const results = [];
    const checkedLinks = new Set();
    await fs.mkdir(path.join(root, 'test-results'), {recursive: true});
    for (const {slug, url, figures} of pages) {
      await page.goto(`${base}${url}`);
      await page.waitForFunction(() => window.MathJax?.startup?.promise, null, {timeout: 30000});
      await page.evaluate(() => MathJax.startup.promise);
      const stats = await page.evaluate(() => ({
        formulas: document.querySelectorAll('.arithmatex').length,
        rendered: document.querySelectorAll('.arithmatex mjx-container[jax="SVG"]').length,
        errors: [...document.querySelectorAll('[data-mjx-error], mjx-merror')].map(n => n.textContent),
      }));
      assert.equal(stats.rendered, stats.formulas, `${slug}: unrendered math`);
      assert.deepEqual(stats.errors, [], `${slug}: TeX errors`);
      await page.locator('article img').evaluateAll(async images => {
        await Promise.all(images.map(img => { img.loading = 'eager'; return img.decode(); }));
      });
      assert.equal(await page.locator('figure').count(), figures, `${slug}: missing figures`);
      assert.equal(await page.locator('figure figcaption').count(), figures, `${slug}: missing captions`);
      for (const link of await page.locator('a[href]').evaluateAll(links => links.map(a => a.href))) {
        if (!link.startsWith(base)) continue;
        if (new URL(link).hash || checkedLinks.has(link)) continue;
        checkedLinks.add(link);
        const response = await page.request.get(link);
        assert.equal(response.status(), 200, `${slug}: broken link ${link}`);
      }
      for (const width of [1440, 390]) {
        await page.setViewportSize({width, height: 1000});
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        assert.ok(overflow <= 1, `${slug}: viewport ${width} overflowed by ${overflow}px`);
      }
      await page.setViewportSize({width: 1440, height: 1000});
      await page.screenshot({path: path.join(root, 'test-results', `${slug}-header.png`)});
      if (figures) {
        await page.locator('figure').first().scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(root, 'test-results', `${slug}-figure.png`)});
      }
      results.push({slug, ...stats});
      console.log(`${slug}: ${stats.rendered} formulas rendered`);
    }
    await page.goto(base);
    for (const width of [1440, 390]) {
      await page.setViewportSize({width, height: 1000});
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `home ${width}: overflow`);
      await page.screenshot({path: path.join(root, 'test-results', `home-${width}.png`)});
    }
    assert.equal(await page.locator('.card').count(), 13);
    await page.locator('#theme-toggle').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    report.results = results;
  } finally {
    try {
      if (browser) await browser.close();
    } finally {
      await new Promise(resolve => server.close(resolve));
    }
  }
  report.status = 'passed'; report.finishedAt = new Date().toISOString();
  await fs.writeFile(output, JSON.stringify(report, null, 2));
}
main().catch(async error => {
  const output = path.join(root, 'test-results', 'browser-math.json');
  try {
    const report = JSON.parse(await fs.readFile(output, 'utf8'));
    report.status = 'failed'; report.error = String(error); report.finishedAt = new Date().toISOString();
    await fs.writeFile(output, JSON.stringify(report, null, 2));
  } catch (writeError) { console.error('Could not record failed browser run:', writeError); }
  console.error(error); process.exitCode = 1;
});
