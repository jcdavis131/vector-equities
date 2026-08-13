/* vector-equities PWA v67 — 74426B HIT void #080A0F — PWA shell-only, CORE immutable stale-while-revalidate, large JSON_ONNX deny-cached
   Mirrors hoops v67 pattern — shell-only lightweight 4831 FYs dark + void #080A0F
   - CACHE v67 74426B (full bundle 74k with CORE list) — HIT = high-intensity trading void dark canvas
   - CORE only shell (19 files), no large JSON/models — 800 FYs LOD drag-map→Jordan
   - network-first for js/css/img assets with 1MB cache cap
   - JSON deliberately never SW-cached (network only, browser HTTP cache still applies)
     => offline mode is shell-only; data pages need connection — free platform, no charging
   - stale-while-revalidate for immutable CORE
   - DENY real_data.json real_pca_full universe_full_history mtnn.onnx data
   - Knowledge→Edge→Money — same-link-same-stars LCG dailySeed 20260812 idx3970 triple[3970,14390,4582]
   - zero-deps true torch auto cuda else cpu
*/

const CACHE_NAME = 'vector-equities-v67-dark';

const CORE = [
  '/',
  '/play',
  '/manifest.json',
  '/offline.html',
  '/assets/shell.css',
  '/assets/responsive.css',
  '/assets/final-qa.css',
  '/assets/unified.css',
  '/assets/motion.css',
  '/assets/player-profile-v28.css',
  '/assets/trading-card.css',
  '/assets/site-nav.js',
  '/assets/error-boundary.js',
  '/assets/keyboard-a11y.js',
  '/assets/pwa-install.js',
  '/assets/og-embed.png',
  '/assets/og-1200x630.png',
  '/assets/icon-192.png',
  '/assets/icon-512.png'
];

const DENY_CACHE = [
  '/assets/vectors.json',
  '/assets/real_data.json',
  '/assets/real_pca_full.json',
  '/assets/real_pca.json',
  '/assets/universe_full_history.json',
  '/assets/universe_full_history_manifest.json',
  '/assets/mtnn.onnx',
  '/assets/mtnn.onnx.data',
  '/assets/mtnn_heads.f32',
  '/assets/mtnn_embeddings.f32',
  '/assets/data/equities.json'
];

const FULL_MTNN = [
  '/assets/mtnn_embeddings.f32',
  '/assets/mtnn_heads.f32',
  '/assets/mtnn_arch.json',
  '/assets/mtnn_meta.json',
  '/assets/mtnn_map.json',
  '/assets/network-viz.js',
  '/assets/mtnn-full.js',
  '/assets/mtnn-worker.js',
  '/assets/mtnn-onnx.js',
  '/assets/vectors_lite.json',
  '/assets/real_data.json',
  '/assets/real_pca_full.json',
  '/assets/skills.json',
  '/assets/universe_full_history.json'
];

function isDenied(p) {
  return DENY_CACHE.some(x => p.includes(x));
}

function isImmutable(url) {
  return CORE.includes(url.pathname);
}

function isAsset(url) {
  const p = url.pathname;
  if (!p.startsWith('/assets/')) return false;
  return (
    p.endsWith('.js') ||
    p.endsWith('.css') ||
    p.endsWith('.png') ||
    p.endsWith('.svg') ||
    p.endsWith('.webp') ||
    p.endsWith('.jpg') ||
    p.endsWith('.jpeg')
  );
}

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    const results = await Promise.allSettled(
      CORE.map((u) => cache.add(new Request(u, { cache: 'reload' })))
    );
    const failed = results.filter(r => r.status === 'rejected');
    if (failed.length) {
      console.warn('[sw v66-dark] CORE precache partial failures:', failed.length);
    }
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    if ('navigationPreload' in self.registration) {
      try {
        await self.registration.navigationPreload.enable();
      } catch {}
    }
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  if (url.origin !== location.origin) return;

  // 1. Denied large assets -> network only, never cache
  if (isDenied(url.pathname)) {
    e.respondWith(
      fetch(req).catch(() => new Response('', { status: 504, statusText: 'Denied asset offline' }))
    );
    return;
  }

  // 2. Navigate -> network first, fallback to cache / offline.html
  const isNavigate = req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html');
  if (isNavigate) {
    e.respondWith((async () => {
      try {
        const preload = await e.preloadResponse;
        if (preload) {
          const c = await caches.open(CACHE_NAME);
          c.put(req, preload.clone()).catch(() => {});
          return preload;
        }
        const net = await fetch(req);
        if (net && net.ok) {
          const c = await caches.open(CACHE_NAME);
          c.put(req, net.clone()).catch(() => {});
        }
        return net;
      } catch {
        const cached = await caches.match(req);
        if (cached) return cached;
        const off = await caches.match('/offline.html');
        if (off) return off;
        return caches.match('/') || new Response('Offline', { status: 503 });
      }
    })());
    return;
  }

  // 3. Immutable CORE -> stale-while-revalidate (instant cache, update bg)
  if (isImmutable(url)) {
    e.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(req);
      const fetchPromise = fetch(req)
        .then((r) => {
          if (r && r.ok) cache.put(req, r.clone()).catch(() => {});
          return r;
        })
        .catch(() => null);
      if (cached) {
        e.waitUntil(fetchPromise);
        return cached;
      }
      const net = await fetchPromise;
      return net || cached || Response.error();
    })());
    return;
  }

  // 4. Asset (js/css/png/svg/webp) -> network-first, cache only if <1MB
  if (isAsset(url)) {
    e.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      try {
        const net = await fetch(req);
        if (net && net.ok) {
          const size = parseInt(net.headers.get('content-length') || '0', 10);
          if (!size || size < 1_000_000) cache.put(req, net.clone()).catch(() => {});
        }
        return net;
      } catch {
        const cached = await cache.match(req);
        if (cached) return cached;
        return new Response('', { status: 504, statusText: 'Asset offline' });
      }
    })());
    return;
  }

  // 5. Everything else (e.g. /assets/*.json not in CORE) -> try cache then network, but JSON never cached by SW (browser cache still ok)
  e.respondWith((async () => {
    try {
      return await fetch(req);
    } catch {
      const cached = await caches.match(req);
      if (cached) return cached;
      return new Response('', { status: 504, statusText: 'Offline' });
    }
  })());
});

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});
