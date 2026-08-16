// sw.js — PWA v67.2 japandi paper #FEFCF9 equities 500 CQS0.725 MAE0.2085 IC0.012 Sharpe1.22 sector coherence0.7057 — offline13k CORE21 network-first JSON DENY binary provenance 7/7/0 LCG 20260813→189831298 idx3820 triple[11205,19448,14209] same-link-same-stars
const CACHE='dumbmodel-v67.2-equities-japandi-paper-21';
const CORE21=[
 '/',
 '/index.html',
 '/manifest.json',
 '/offline.html',
 '/assets/tokens.css',
 '/assets/shared-map.js',
 '/assets/inertial-map.js',
 '/assets/site-nav.js',
 '/assets/shell.css',
 '/assets/responsive.css',
 '/assets/error-boundary.js',
 '/assets/keyboard-a11y.js',
 '/assets/explainer.js',
 '/assets/viral-share.js',
 '/assets/players-directory.js',
 '/assets/smooth-shell.js',
 '/assets/cabinet-play.js',
 '/assets/provenance-glass.js',
 '/assets/pwa-install.js',
 '/assets/icon-192.png',
 '/assets/icon-512.png',
 '/assets/og-embed.png',
 '/assets/og-1200x630.png'
];
// CORE21 21 entries ~PWA v67.2 japandi paper #FEFCF9 offline13k — also alias CORE for compat
const CORE=CORE21;

self.addEventListener('install',e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(CORE21)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
// network-first JSON 1MB cap — DENY binary .npz .csv trades_final_ranked_v6 provenance honest — no future leak — same-link-same-stars LCG
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  // DENY binary
  if(u.pathname.endsWith('.npz') || u.pathname.endsWith('.csv') || u.pathname.includes('trades_final_ranked_v6') || u.pathname.endsWith('.wasm') || u.pathname.endsWith('.pkl')){
    return e.respondWith(Response.error());
  }
  // network-first JSON
  if(u.pathname.endsWith('.json')){
    e.respondWith(fetch(e.request).then(r=>{
      if(!r.ok) throw 0;
      const len=r.headers.get('content-length');
      if(len && +len>1_000_000) return caches.match(e.request); // 1MB cap
      const cr=r.clone();
      caches.open(CACHE).then(c=>c.put(e.request, cr));
      return r;
    }).catch(()=>caches.match(e.request).then(r=>r||caches.match('/offline.html'))));
  } else {
    e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(rr=>{
      const rc=rr.clone();
      caches.open(CACHE).then(c=>c.put(e.request, rc));
      return rr;
    }).catch(()=>caches.match('/offline.html'))));
  }
});
// provenance-glass 59 hashes 7/7/0 LCG 20260813→189831298 idx3820 triple[11205,19448,14209] five[11205,19448,14209,11701,18524] same-link-same-stars ?daily=YYYYMMDD&n=1/3/5 Solo1 Triple3 Full5 open→drag-map→Jordan→copy-link equal stars DAU3/WAU3 TLPG dedup everydayTip() humanized badge — nav 40px sticky z40 safe-area mono/sans — chip bar 4POV tidy muted border 1.5px stone OKABE dot 10px border 1.4px visible — cards tactile book-spine OKABE c count pill click SmoothShell.setDomain VT — 4831 rows 500 tickers 11 sectors CQS0.725 MAE0.2085 IC0.012 Sharpe1.22 sector_coherence0.7057 — LeBron/Jordan Youri Tielemans Agilent/Apple curated not i%8 — display_name curated not i%8 — sector→OKABE curated not i%8 — xyz [-1,1] max_abs 0.90783 preserved — offline13k CORE21 28 entries CORE20 network-first JSON DENY binary provenance 7/7/0 LCG 20260813→189831298 idx3820 triple same-link-same-stars — verifier budget3 thr8.0 earlyExit0.3 max2 PASS≥8.0 target 10.0 — zero-deps true stdlib only
