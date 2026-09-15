/* ===========================================================
 * sw.js — whiteicey.github.io
 * Network-first caching with offline fallback.
 * ========================================================== */

const PRECACHE = 'precache-v3';
const RUNTIME = 'runtime-v3';
const HOSTNAME_WHITELIST = [
  self.location.hostname,
  'whiteicey.github.io',
  'cdnjs.cloudflare.com'
];

const isNavigationRequest = (request) => {
  const accept = request.headers.get('accept') || '';
  return request.mode === 'navigate' || (request.method === 'GET' && accept.includes('text/html'));
};

const hasExtension = (request) => Boolean(new URL(request.url).pathname.match(/\.\w+$/));

const shouldRedirect = (request) => (
  isNavigationRequest(request) &&
  !new URL(request.url).pathname.endsWith('/') &&
  !hasExtension(request)
);

const getRedirectUrl = (request) => {
  const url = new URL(request.url);
  url.pathname += '/';
  return url.href;
};

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(PRECACHE)
      .then((cache) => cache.add('offline.html'))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((key) => {
      if (key !== PRECACHE && key !== RUNTIME) {
        return caches.delete(key);
      }
      return undefined;
    }));
    await self.clients.claim();
  })());
});

async function handleFetch(event) {
  try {
    // Always ask the network first. This prevents a deployed UI from rendering
    // with a stale CSS/JS bundle until the user manually refreshes.
    const response = await fetch(event.request, { cache: 'no-store' });

    if (response && response.ok) {
      try {
        const cache = await caches.open(RUNTIME);
        await cache.put(event.request, response.clone());
      } catch (cacheError) {
        // Cross-origin or unsupported responses may not be cacheable.
      }
    }

    return response;
  } catch (networkError) {
    const cached = await caches.match(event.request, { ignoreSearch: true });
    if (cached) return cached;

    if (isNavigationRequest(event.request)) {
      const offline = await caches.match('offline.html');
      if (offline) return offline;
    }

    return new Response('Offline', {
      status: 503,
      statusText: 'Offline',
      headers: { 'Content-Type': 'text/plain; charset=utf-8' }
    });
  }
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  let hostname;
  try {
    hostname = new URL(event.request.url).hostname;
  } catch (error) {
    return;
  }

  if (!HOSTNAME_WHITELIST.includes(hostname)) return;

  if (shouldRedirect(event.request)) {
    event.respondWith(Response.redirect(getRedirectUrl(event.request)));
    return;
  }

  event.respondWith(handleFetch(event));
});