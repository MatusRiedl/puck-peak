"""Centralised NHL HTTP client with caching, rate limiting, retry, and dedup.

All outbound API traffic should flow through ``get_client().get(…)``.
See ``cache_strategy.md`` §3 for the full design rationale.
"""

import logging
import os
import random
import threading
import time
import urllib.parse
from concurrent.futures import Future

import requests

from nhl.cache import get_cache

log = logging.getLogger("nhl.api")

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT = 10
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_NON_RETRYABLE_STATUS = frozenset({400, 403, 404})

_DOMAIN_RATE_DEFAULTS: dict[str, tuple[float, float]] = {
    "api-web.nhle.com": (20, 1.0),
    "api.nhle.com": (15, 1.0),
    "records.nhl.com": (10, 1.0),
    "search.d3.nhle.com": (5, 1.0),
}
_DEFAULT_RATE: tuple[float, float] = (5, 1.0)


# ---------------------------------------------------------------------------
# TokenBucket — per-domain rate limiter
# ---------------------------------------------------------------------------
class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Args:
        rate:  Number of tokens granted per *per* seconds.
        per:   Time window in seconds.
        burst: Maximum token capacity.  Defaults to ``int(rate * 2)``.
    """

    def __init__(self, rate: float, per: float = 1.0, burst: int | None = None):
        self.rate = rate
        self.per = per
        self.burst = burst if burst is not None else int(rate * 2)
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * (self.rate / self.per))
        self._last_refill = now

    def acquire(self, timeout: float = 10.0) -> bool:
        """Block until a token is available or *timeout* seconds elapse."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # How long until one token is available?
                wait = (1.0 - self._tokens) * (self.per / self.rate)
            # Sleep outside the lock
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(wait, remaining))


# ---------------------------------------------------------------------------
# NHLClient
# ---------------------------------------------------------------------------
class NHLClient:
    """Singleton HTTP client for all NHL API access.

    Features:
    - ``requests.Session`` with connection pooling.
    - Per-domain token-bucket rate limiting.
    - Retry with exponential backoff + jitter on transient errors.
    - Request deduplication for identical in-flight cache keys.
    - Integration with the shared ``NHLCache`` (diskcache / dict fallback).
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "PuckPeak/1.0",
            "Accept": "application/json",
        })

        self._cache = get_cache()

        # Rate limiters -------------------------------------------------------
        self._rate_limiters: dict[str, TokenBucket] = {}
        for domain, (rate, per) in _DOMAIN_RATE_DEFAULTS.items():
            env_key = "NHL_RATE_LIMIT_" + domain.replace(".", "_").replace("-", "_").upper()
            rate = float(os.environ.get(env_key, rate))
            self._rate_limiters[domain] = TokenBucket(rate, per)

        # In-flight deduplication ----------------------------------------------
        self._inflight: dict[str, Future] = {}
        self._inflight_lock = threading.Lock()

        # Retry config ---------------------------------------------------------
        self._max_retries = int(os.environ.get("NHL_MAX_RETRIES", _MAX_RETRIES))
        self._retry_base_delay = float(
            os.environ.get("NHL_RETRY_BASE_DELAY", _RETRY_BASE_DELAY)
        )

        self._log = log

    # -- public API -----------------------------------------------------------

    def get(
        self,
        url: str,
        params: dict | None = None,
        cache_key: str | None = None,
        ttl: int | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> dict | None:
        """Fetch JSON from *url*, with caching / dedup / retry / rate-limit.

        Returns parsed JSON ``dict`` on success or ``None`` on failure.
        When *cache_key* is provided the result is stored in (and served from)
        the shared disk cache with the given *ttl* in seconds.
        """
        # 1. Cache lookup ------------------------------------------------------
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._log.debug("Cache hit: %s", cache_key)
                return cached

        # 2. Deduplication — only when cache_key is set -----------------------
        existing_future: Future | None = None
        my_future: Future | None = None

        if cache_key is not None:
            with self._inflight_lock:
                if cache_key in self._inflight:
                    existing_future = self._inflight[cache_key]
                else:
                    my_future = Future()
                    self._inflight[cache_key] = my_future

        # Await an existing in-flight request
        if existing_future is not None:
            self._log.debug("Dedup await: %s", cache_key)
            try:
                return existing_future.result(timeout=30)
            except Exception:
                return None

        # 3. We own the request — execute it ---------------------------------
        try:
            # Rate-limit
            limiter = self._get_rate_limiter(url)
            if not limiter.acquire(timeout=10.0):
                self._log.warning("Rate-limit timeout for %s", url)

            result = self._do_request(url, params, timeout)

            # Cache store
            if result is not None and cache_key is not None:
                self._cache.set(cache_key, result, ttl=ttl)

            # Resolve future
            if my_future is not None:
                my_future.set_result(result)

            return result

        except Exception as exc:
            self._log.error("Unhandled error for %s: %s", url, exc)
            if my_future is not None and not my_future.done():
                my_future.set_result(None)
            return None

        finally:
            if cache_key is not None and my_future is not None:
                with self._inflight_lock:
                    self._inflight.pop(cache_key, None)

    def invalidate(self, pattern: str) -> int:
        """Remove cache entries whose key matches a fnmatch *pattern*."""
        return self._cache.invalidate(pattern)

    # -- internals ------------------------------------------------------------

    def _do_request(
        self, url: str, params: dict | None, timeout: float
    ) -> dict | None:
        """Execute GET with retry + backoff.  Returns parsed JSON or None."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=timeout)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code in _NON_RETRYABLE_STATUS:
                    self._log.warning(
                        "Non-retryable %d from %s", resp.status_code, url
                    )
                    return None

                if resp.status_code in _RETRYABLE_STATUS:
                    delay = self._backoff_delay(attempt, resp)
                    self._log.warning(
                        "%d from %s — retry %d/%d in %.1fs",
                        resp.status_code, url, attempt + 1, self._max_retries, delay,
                    )
                    time.sleep(delay)
                    continue

                # Unexpected status
                self._log.warning("Unexpected %d from %s", resp.status_code, url)
                return None

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                delay = self._backoff_delay(attempt)
                self._log.warning(
                    "%s for %s — retry %d/%d in %.1fs",
                    type(exc).__name__, url, attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)

            except Exception as exc:
                self._log.error("Request error for %s: %s", url, exc)
                return None

        self._log.error(
            "All %d retries exhausted for %s (last: %s)",
            self._max_retries, url, last_exc,
        )
        return None

    def _backoff_delay(self, attempt: int, resp=None) -> float:
        """Compute retry delay with exponential backoff, jitter, and 429 awareness."""
        base = self._retry_base_delay * (2 ** attempt)
        jitter = random.uniform(-0.5, 0.5)
        delay = max(0.1, base + jitter)

        # Respect Retry-After on 429
        if resp is not None and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            else:
                delay = max(delay, 5.0)

        return delay

    def _get_rate_limiter(self, url: str) -> TokenBucket:
        """Return the rate-limiter bucket for *url*'s domain."""
        domain = urllib.parse.urlparse(url).hostname or ""
        limiter = self._rate_limiters.get(domain)
        if limiter is None:
            limiter = TokenBucket(*_DEFAULT_RATE)
            self._rate_limiters[domain] = limiter
        return limiter


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_client: NHLClient | None = None
_client_lock = threading.Lock()


def get_client() -> NHLClient:
    """Return the shared ``NHLClient`` singleton (lazy-initialised)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        _client = NHLClient()
        return _client
