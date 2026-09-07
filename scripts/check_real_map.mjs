// Run the page's real load->transform->centroid->join path against the actual JSON
// files, outside a browser. This is the validate gate: it fails loudly rather than
// letting a page ship that silently draws nothing.
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..', 'public');
const read = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), 'utf-8'));

let fails = 0;
const ok = (cond, msg) => {
  console.log((cond ? '  PASS  ' : '  FAIL  ') + msg);
  if (!cond) fails++;
};

const map = read('assets/map_slim.json');
const ros = read('assets/roster.json');

console.log('\n-- file contract --');
ok(Array.isArray(map.rows), 'map_slim.json has a rows array');
ok(map.rows.length === map.n_rows, `rows ${map.rows.length} == declared n_rows ${map.n_rows}`);
ok(map.rows.length === 4831, 'row count is the full 4,831 filing-years');
ok(Array.isArray(map.sectors) && map.sectors.length === 11, '11 sectors declared');
ok(Array.isArray(ros.tiles) && ros.tiles.length === 11, '11 roster tiles, one per sector');

console.log('\n-- no fabricated values --');
ok(map.rows.every(r => Number.isFinite(r.x) && Number.isFinite(r.y)),
   'every row has finite coordinates');
const distinctX = new Set(map.rows.map(r => r.x)).size;
ok(distinctX > 4000, `x is genuinely varied, not generated from a few centres (${distinctX} distinct)`);
ok(map.rows.every(r => r.ticker && r.name && r.sector && r.year),
   'every row carries ticker, name, sector and year');
ok(!JSON.stringify(map).includes('offense') && !JSON.stringify(ros).includes('offense'),
   'no basketball vocabulary survives in the data files');
ok(!JSON.stringify(ros).includes('archetype'),
   'the roster does not carry the invented archetype names');

console.log('\n-- the view transform the page applies --');
let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
for (const r of map.rows) {
  if (r.x < x0) x0 = r.x; if (r.x > x1) x1 = r.x;
  if (r.y < y0) y0 = r.y; if (r.y > y1) y1 = r.y;
}
const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
const points = map.rows.map(r => ({
  x: (r.x - x0) / dx, y: 1 - (r.y - y0) / dy,
  ticker: r.ticker, year: r.year, sector: r.sector, name: r.name,
}));
ok(points.every(p => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1),
   'every transformed point lands inside the 0..1 canvas box');
const spanX = Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x));
ok(Math.abs(spanX - 1) < 1e-9, 'the transform uses the full width (no dead margin)');

console.log('\n-- sector centroids, as the page computes them --');
const acc = new Map();
for (const p of points) {
  const a = acc.get(p.sector) || { sx: 0, sy: 0, n: 0 };
  a.sx += p.x; a.sy += p.y; a.n++;
  acc.set(p.sector, a);
}
const centers = [...acc.entries()]
  .map(([sector, a]) => ({ sector, x: a.sx / a.n, y: a.sy / a.n, n: a.n }))
  .sort((p, q) => q.n - p.n);
ok(centers.length === 11, '11 centroids computed');
const top3 = centers.slice(0, 3).map(c => c.sector);
ok(['Industrials', 'Financials', 'Technology'].every(s => top3.includes(s)),
   `the three spotlight steps are the three largest sectors (${top3.join(', ')})`);

console.log('\n  centroid table (what the prose must match):');
for (const c of centers) {
  console.log(`    ${c.sector.padEnd(24)} x ${c.x.toFixed(3)}  y ${c.y.toFixed(3)}  n ${String(c.n).padStart(4)}`);
}

console.log('\n-- prose claims checked against those numbers --');
const cx = centers.map(c => c.x);
const centroidSpan = Math.max(...cx) - Math.min(...cx);
ok(Math.abs(centroidSpan - 0.23) < 0.02,
   `centroids span ~0.23 of the width as the prose states (actual ${centroidSpan.toFixed(3)})`);
const fin = centers.find(c => c.sector === 'Financials');
ok(fin.x === Math.min(...cx) || Math.abs(fin.x - Math.min(...cx)) < 0.02,
   'Financials is at or beside the leftmost centroid, as step 5 states');
ok(fin.y === Math.max(...centers.map(c => c.y)),
   'Financials is the lowest centroid, as step 5 states');
const tech = centers.find(c => c.sector === 'Technology');
const mid = (Math.max(...cx) + Math.min(...cx)) / 2;
ok(Math.abs(tech.x - mid) < 0.08, 'Technology sits mid-map, as step 6 states');
const ind = centers.find(c => c.sector === 'Industrials');
ok(ind.n === Math.max(...centers.map(c => c.n)) && ind.n === 768,
   'Industrials is the largest sector at 768 filing-years, as step 4 states');
ok(fin.n === 740 && tech.n === 707, 'Financials 740 and Technology 707, as steps 5 and 6 state');

console.log('\n-- roster tiles join to real map points --');
for (const t of ros.tiles) {
  const hit = points.find(p => p.ticker === t.ticker && p.year === t.year);
  ok(!!hit, `${t.ticker} FY${t.year} (${t.sector}) is findable on the map`);
  if (hit) {
    ok(hit.sector === t.sector, `  ${t.ticker} sector agrees between roster and map`);
  }
}
ok(ros.tiles.every(t => (t.top_skills || []).length === 2),
   'every tile has two named skills');
ok(ros.tiles.every(t => t.top_skills.every(s => ros.skill_keys.includes(s.key))),
   'every skill name comes from the export\'s own skill_keys list');

console.log('\n-- palette --');
const SET3 = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462',
              '#b3de69','#fccde5','#ffed6f','#bc80bd','#ccebc5'];
ok(new Set(SET3).size === 11, '11 distinct sector colours');
ok(!SET3.includes('#BFC2CC') && !SET3.includes('#d9d9d9'),
   'no sector colour collides with the gray context dots');

console.log(fails === 0
  ? '\nALL CHECKS PASSED\n'
  : `\n${fails} CHECK(S) FAILED\n`);
process.exit(fails === 0 ? 0 : 1);
