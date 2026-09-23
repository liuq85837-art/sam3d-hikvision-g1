const { chromium } = require('C:/Users/liuq8/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

(async () => {
  const dir = __dirname;
  const manual = path.join(dir, '..', 'SAM3D_零基础完整说明书.html');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [];
  const network = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', req => { if (/^https?:/.test(req.url())) network.push(req.url()); });
  await page.goto(pathToFileURL(manual).href);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: path.join(dir, 'manual_desktop_top.png') });
  for (const [id, name] of [['s05', 'video'], ['s07', 'commands'], ['s13', 'g1'], ['s22', 'ending']]) {
    await page.locator(`#${id}`).evaluate(el => el.scrollIntoView({ behavior: 'instant', block: 'start' }));
    await page.screenshot({ path: path.join(dir, `manual_desktop_${name}.png`) });
  }
  const desktop = await page.evaluate(() => ({
    title: document.title,
    chapters: [...document.querySelectorAll('a[id]')].filter(el => /^s\d+$/.test(el.id)).length,
    headings: document.querySelectorAll('h2').length,
    codeBlocks: document.querySelectorAll('pre').length,
    tables: document.querySelectorAll('table').length,
    imagesLoaded: [...document.images].every(im => im.complete && im.naturalWidth > 0),
    brokenAnchors: [...document.querySelectorAll('a[href^="#"]')].map(el => el.getAttribute('href').slice(1)).filter(id => id && !document.getElementById(decodeURIComponent(id))),
    pageOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
  }));
  await page.setViewportSize({ width: 430, height: 932 });
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: path.join(dir, 'manual_mobile_top.png') });
  const mobile = await page.evaluate(() => ({ pageOverflow: document.documentElement.scrollWidth > window.innerWidth + 1 }));
  const report = { desktop, mobile, errors, externalRequests: network };
  report.passed = desktop.chapters === 22 && desktop.imagesLoaded && !desktop.pageOverflow && !mobile.pageOverflow && desktop.brokenAnchors.length === 0 && errors.length === 0 && network.length === 0;
  fs.writeFileSync(path.join(dir, 'manual_browser_qa.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
  if (!report.passed) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
