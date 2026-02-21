"""
Cache Service Adapter - Professional Implementation

Provides institutional-grade cache service adapter that bridges MultiTTLCache
with the ICacheService interface for proper dependency injection.
"""

from typing import Optional, Any, Dict
from services.interfaces import ICacheService
from utils.ttl_cache import MultiTTLCache


class CacheServiceAdapter(ICacheService):
    """
    Professional adapter that implements ICacheService interface
    using MultiTTLCache as the underlying cache implementation.

    This adapter enables proper dependency injection while maintaining
    the advanced TTL and performance features of MultiTTLCache.
    """

    def __init__(self, default_cache_type: str = "default", default_ttl: float = 3600.0):
        """
        Initialize cache adapter with institutional-grade defaults.

        Args:
            default_cache_type: Default cache namespace for operations
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.cache = MultiTTLCache()
        self.default_cache_type = default_cache_type
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        return self.cache.get(self.default_cache_type, key)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set cached value with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self.cache.set(self.default_cache_type, key, value, effective_ttl)

    def delete(self, key: str) -> bool:
        """
        Delete cached value by key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        return self.cache.delete(self.default_cache_type, key)

    def clear(self) -> None:
        """Clear all cached values in the default cache type."""
        self.cache.clear(self.default_cache_type)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        stats = self.cache.get_stats()
        return {
            'total_entries': stats.get('total_entries', 0),
            'total_hits': stats.get('total_hits', 0),
            'total_misses': stats.get('total_misses', 0),
            'hit_rate': stats.get('hit_rate', 0.0),
            'cache_types': len(stats.get('cache_types', [])),
            'memory_usage_mb': stats.get('memory_usage_mb', 0.0)
        }

    # Extended methods for advanced cache operations
    def get_with_cache_type(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from specific cache type namespace."""
        return self.cache.get(cache_type, key)

    def set_with_cache_type(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in specific cache type namespace."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self.cache.set(cache_type, key, value, effective_ttl)

    def clear_cache_type(self, cache_type: str) -> None:
        """Clear all values from specific cache type."""
        self.cache.clear(cache_type)

    def get_cache_types(self) -> list:
        """Get list of all active cache type namespaces."""
        stats = self.cache.get_stats()
        return list(stats.get('cache_types', []))