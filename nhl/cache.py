"""Shared cache layer backed by diskcache (SQLite) with a dict fallback.

Provides TTL-tier constants matching the data taxonomy in cache_strategy.md
and a singleton ``get_cache()`` accessor used by ``nhl.api.NHLClient``.
"""

import fnmatch
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from nhl.constants import CURRENT_SEASON_YEAR

log = logging.getLogger("nhl.cache")

# ---------------------------------------------------------------------------
# TTL tier constants (seconds)
# ---------------------------------------------------------------------------
T0_TTL = None       # permanent / process lifetime
T1_TTL = 86_400     # 24 hours  — historical / immutable
T2_DEFAULT_TTL = 3_600   # 1 hour — semi-static seasonal
T3_DEFAULT_TTL = 120     # 2 minutes — live / near-real-time

_MISSING = object()


def effective_ttl(season_year: int) -> int:
    """Return T1 if *season_year* is a closed season, else T2 default.

    Many NHL endpoints accept a season year.  Past-season data is immutable
    (T1 / 24 h) while current-season data refreshes hourly (T2 / 1 h).
    """
    if season_year < CURRENT_SEASON_YEAR:
        return T1_TTL
    return T2_DEFAULT_TTL


# ---------------------------------------------------------------------------
# _DictCache — bounded in-process fallback
# ---------------------------------------------------------------------------
class _DictCache:
    """Bounded dict-based cache with TTL, used when *diskcache* is absent."""

    def __init__(self, maxsize: int = 500):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # -- public API ----------------------------------------------------------

    def get(self, key: str):
        with self._lock:
            entry = self._data.get(key, _MISSING)
            if entry is _MISSING:
                self._misses += 1
                return _MISSING
            value, expiry = entry
            if expiry is not None and time.monotonic() > expiry:
                del self._data[key]
                self._misses += 1
                return _MISSING
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value, ttl=None):
        expiry = (time.monotonic() + ttl) if ttl is not None else None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, expiry)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def invalidate(self, pattern: str) -> int:
        """Remove all entries whose key matches *pattern* (fnmatch glob)."""
        with self._lock:
            keys = [k for k in self._data if fnmatch.fnmatch(k, pattern)]
            for k in keys:
                del self._data[k]
            return len(keys)

    def close(self):
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses}


# ---------------------------------------------------------------------------
# NHLCache — primary cache interface
# ---------------------------------------------------------------------------
class NHLCache:
    """Wraps *diskcache.Cache* or falls back to ``_DictCache``.

    Callers should use the module-level ``get_cache()`` singleton instead of
    instantiating this directly.
    """

    def __init__(self, directory: str | Path, size_limit: int):
        self._backend_name: str
        try:
            import diskcache
            self._store = diskcache.Cache(
                directory=str(directory),
                size_limit=size_limit,
                eviction_policy="least-recently-used",
                disk_min_file_size=0,
                statistics=True,
            )
            self._backend_name = "diskcache"
            log.info("Cache initialised with diskcache at %s", directory)
        except ImportError:
            log.warning(
                "diskcache not installed — falling back to in-process dict "
                "cache.  pip install diskcache for cross-process persistence."
            )
            self._store = _DictCache()
            self._backend_name = "dict"

    # -- public API ----------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def get(self, key: str):
        """Return cached value or ``None`` if not found / expired."""
        if self._backend_name == "diskcache":
            result = self._store.get(key, default=_MISSING)
        else:
            result = self._store.get(key)
        if result is _MISSING:
            return None
        return result

    def set(self, key: str, value, ttl=None):
        """Store *value* under *key*.  *ttl* is seconds or ``None``."""
        if self._backend_name == "diskcache":
            self._store.set(key, value, expire=ttl)
        else:
            self._store.set(key, value, ttl=ttl)

    def delete(self, key: str) -> bool:
        if self._backend_name == "diskcache":
            return self._store.delete(key)
        return self._store.delete(key)

    def invalidate(self, pattern: str) -> int:
        """Delete all entries whose key matches a fnmatch *pattern*."""
        if self._backend_name == "diskcache":
            count = 0
            for key in list(self._store):
                if fnmatch.fnmatch(key, pattern):
                    self._store.delete(key)
                    count += 1
            return count
        return self._store.invalidate(pattern)

    def close(self):
        self._store.close()

    def stats(self) -> dict:
        if self._backend_name == "diskcache":
            hits, misses = self._store.stats()
            return {"hits": hits, "misses": misses}
        return self._store.stats()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_cache: NHLCache | None = None
_cache_lock = threading.Lock()

_DEFAULT_DIR = ".cache/nhl_api"
_DEFAULT_SIZE_LIMIT = 200 * 1024 * 1024  # 200 MB


def get_cache() -> NHLCache:
    """Return the shared ``NHLCache`` singleton (lazy-initialised)."""
    global _default_cache
    if _default_cache is not None:
        return _default_cache
    with _cache_lock:
        if _default_cache is not None:          # double-check
            return _default_cache
        directory = Path(os.environ.get("NHL_CACHE_DIR", _DEFAULT_DIR))
        _default_cache = NHLCache(directory, _DEFAULT_SIZE_LIMIT)
        return _default_cache
