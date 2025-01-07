"""
# ancilla/utils/caching.py
Flexible caching system with both memory and compressed file-based caching capabilities.
"""

import gzip
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Union, Dict, Callable
from functools import wraps
import threading


class CacheBase:
    """Base class defining the caching interface."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class MemoryCache(CacheBase):
    """Thread-safe in-memory cache with TTL."""

    def __init__(self, ttl: int = 300):
        """
        Initialize memory cache.

        Args:
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if (datetime.now() - entry["timestamp"]).total_seconds() > self._ttl:
                del self._cache[key]
                return None

            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = {"value": value, "timestamp": datetime.now()}

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class FileCache(CacheBase):
    """Compressed file-based cache with TTL."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".ancilla_cache",
        ttl: int = 86400,
        cleanup_interval: int = 3600,
    ):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds (default: 24 hours)
            cleanup_interval: Interval for cleanup tasks in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self._lock = threading.RLock()
        self._last_cleanup = datetime.now()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Start cleanup thread
        self._start_cleanup_thread()

    def _get_cache_path(self, key: str) -> Path:
        """Generate a cache file path from a key."""
        # Use MD5 for filename to handle long keys
        filename = hashlib.md5(key.encode()).hexdigest() + ".cache.gz"
        return self.cache_dir / filename

    def _save_to_file(self, path: Path, data: Any) -> None:
        """Save data to a compressed file."""
        with gzip.open(path, "wb") as f:
            pickle.dump(
                {"timestamp": datetime.now(), "data": data},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def _load_from_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load data from a compressed file."""
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.PickleError):
            return None

    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if a cache entry has expired."""
        return (datetime.now() - timestamp).total_seconds() > self.ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            path = self._get_cache_path(key)
            if not path.exists():
                return None

            cache_data = self._load_from_file(path)
            if cache_data is None:
                return None

            if self._is_expired(cache_data["timestamp"]):
                self.delete(key)
                return None

            return cache_data["data"]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            path = self._get_cache_path(key)
            self._save_to_file(path, value)

    def delete(self, key: str) -> None:
        with self._lock:
            path = self._get_cache_path(key)
            if path.exists():
                path.unlink()

    def clear(self) -> None:
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache.gz"):
                cache_file.unlink()

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache.gz"):
                try:
                    cache_data = self._load_from_file(cache_file)
                    if cache_data and self._is_expired(cache_data["timestamp"]):
                        cache_file.unlink()
                except Exception:
                    # If we can't read the file, remove it
                    cache_file.unlink()

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""

        def cleanup_task():
            while True:
                # Sleep first to avoid immediate cleanup on initialization
                threading.Event().wait(self.cleanup_interval)
                try:
                    self._cleanup_expired()
                except Exception:
                    # Log but don't crash the cleanup thread
                    logging.exception("Error during cache cleanup")

        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()


class HybridCache:
    """Combined memory and file cache system."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".ancilla_cache",
        memory_ttl: int = 300,
        file_ttl: int = 86400,
        cleanup_interval: int = 3600,
    ):
        """
        Initialize multi-level cache.

        Args:
            cache_dir: Directory for file cache
            memory_ttl: Memory cache TTL in seconds (default: 5 minutes)
            file_ttl: File cache TTL in seconds (default: 24 hours)
            cleanup_interval: Cleanup interval in seconds (default: 1 hour)
        """
        self.memory_cache = MemoryCache(ttl=memory_ttl)
        self.file_cache = FileCache(
            cache_dir=cache_dir, ttl=file_ttl, cleanup_interval=cleanup_interval
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, trying memory first then file."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try file cache
        value = self.file_cache.get(key)
        if value is not None:
            # Update memory cache
            self.memory_cache.set(key, value)
            return value

        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in both memory and file cache."""
        self.memory_cache.set(key, value)
        self.file_cache.set(key, value)

    def delete(self, key: str) -> None:
        """Delete value from both caches."""
        self.memory_cache.delete(key)
        self.file_cache.delete(key)

    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.file_cache.clear()


def cached(
    func: Optional[Callable] = None, *, key_prefix: str = "", ttl: int = 300
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Cache TTL in seconds

    Example:
        @cached(key_prefix="daily_bars", ttl=3600)
        def get_daily_bars(ticker: str, start_date: str) -> pd.DataFrame:
            ...
    """

    def decorator(f: Callable) -> Callable:
        cache = MemoryCache(ttl=ttl)

        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function name, args, and kwargs
            key_parts = [key_prefix or f.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = f(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result)

            return result

        return wrapper

    return decorator if func is None else decorator(func)
