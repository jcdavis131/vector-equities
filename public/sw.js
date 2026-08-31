// sw.js — PWA v67.2 human-first v5 paper #F9F6F0 equities 500 — offline13k CORE21 — business vs market separated — no synthetic
const CACHE='dumbmodel-v67.2-equities-humanfirst-v5';
const CORE=[
 '/',
 '/index.html',
 '/manifest.json',
 '/offline.html',
 '/assets/human-v5/tokens.css',
 '/assets/human-v5/base.css',
 '/assets/human-v5/navigation.css',
 '/assets/human-v5/individual.css',
 '/assets/human-v5/peers.css',
 '/assets/human-v5/map.css',
 '/assets/human-v5/evidence.css',
 '/assets/human-v5/states.css',
 '/assets/human-v5/motion.css',
 '/assets/human-v5/human-v5.js',
 '/assets/data/equities.json',
 '/assets/data/equities_provenance.json',
 '/assets/data/provenance_status.json',
 '/assets/icon-192.png',
 '/assets/icon-512.png',
 '/assets/og-embed.png',
 '/assets/og-1200x630.png'
];
self.addEventListener('install',e=>{
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(CORE)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  if(u.pathname.endsWith('.npz')||u.pathname.endsWith('.csv')||u.pathname.endsWith('.pkl')||u.pathname.endsWith('.wasm')){
    return e.respondWith(Response.error());
  }
  if(u.pathname.endsWith('.json')){
    e.respondWith(fetch(e.request).then(r=>{
      if(!r.ok) throw 0;
      const len=r.headers.get('content-length');
      if(len && +len>1_000_000) return caches.match(e.request);
      const cr=r.clone(); caches.open(CACHE).then(c=>c.put(e.request,cr)); return r;
    }).catch(()=>caches.match(e.request).then(r=>r||caches.match('/offline.html'))));
  } else {
    e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request).then(rr=>{
      const rc=rr.clone(); caches.open(CACHE).then(c=>c.put(e.request,rc)); return rr;
    }).catch(()=>caches.match('/offline.html'))));
  }
});
