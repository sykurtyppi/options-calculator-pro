"""
High-performance TTL (Time-To-Live) Cache with bounded memory usage.

This cache prevents memory leaks during intensive calendar spread scanning
by implementing size limits and automatic cleanup of expired entries.
"""

import time
import threading
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from collections import OrderedDict
import logging


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    value: Any
    created_at: float
    ttl: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() > (self.created_at + self.ttl)
    
    @property
    def age(self) -> float:
        """Age of entry in seconds"""
        return time.time() - self.created_at
    
    @property
    def time_until_expiry(self) -> float:
        """Time until expiry in seconds"""
        return max(0, (self.created_at + self.ttl) - time.time())


class TTLCache:
    """
    Thread-safe TTL cache with bounded memory usage.
    
    Features:
    - Maximum size limit prevents memory exhaustion
    - TTL-based expiry removes stale data
    - LRU eviction for memory pressure
    - Thread-safe operations
    - Configurable cleanup intervals
    - Cache statistics for monitoring
    """
    
    def __init__(
        self,
        maxsize: int = 10000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
        enable_stats: bool = True
    ):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_stats = enable_stats
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'expirations': 0,
            'cleanups': 0
        }
        
        # Background cleanup
        self._cleanup_timer: Optional[threading.Timer] = None
        self._running = True
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup timer
        self._schedule_cleanup()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Returns None if key doesn't exist or has expired.
        Updates LRU order on access.
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._record_stat('misses')
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key, 'expired')
                self._record_stat('misses')
                self._record_stat('expirations')
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._record_stat('hits')
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache with optional custom TTL.
        
        If cache is at capacity, removes least recently used items.
        """
        if ttl is None:
            ttl = self.default_ttl
        
        with self._lock:
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl
            )
            
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Check capacity before adding new entry
                self._ensure_capacity()
                self._cache[key] = entry
            
            self._record_stat('sets')
    
    def delete(self, key: str) -> bool:
        """Delete specific key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._record_stat('deletes')
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            if self.enable_stats:
                # Reset stats except for cumulative counters
                hits = self._stats['hits']
                misses = self._stats['misses']
                self._stats.clear()
                self._stats.update({
                    'hits': hits,
                    'misses': misses,
                    'sets': 0,
                    'deletes': 0,
                    'evictions': 0,
                    'expirations': 0,
                    'cleanups': 0
                })
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns number of entries removed.
        """
        removed_count = 0
        current_time = time.time()
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time > (entry.created_at + entry.ttl)
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            
            if removed_count > 0:
                self._record_stat('expirations', removed_count)
                self.logger.debug(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed maximum size"""
        while len(self._cache) >= self.maxsize:
            # Remove least recently used item
            lru_key = next(iter(self._cache))
            del self._cache[lru_key]
            self._record_stat('evictions')
    
    def _remove_entry(self, key: str, reason: str) -> None:
        """Remove entry and log reason"""
        if key in self._cache:
            del self._cache[key]
            self.logger.debug(f"Removed cache entry '{key}' ({reason})")
    
    def _record_stat(self, stat_name: str, count: int = 1) -> None:
        """Record cache statistics"""
        if self.enable_stats:
            self._stats[stat_name] += count
    
    def _schedule_cleanup(self) -> None:
        """Schedule next cleanup operation"""
        if not self._running:
            return
        
        def cleanup_task():
            if self._running:
                self.cleanup_expired()
                self._record_stat('cleanups')
                self._schedule_cleanup()  # Schedule next cleanup
        
        self._cleanup_timer = threading.Timer(self.cleanup_interval, cleanup_task)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def stop(self) -> None:
        """Stop background cleanup and close cache"""
        self._running = False
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        self.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Useful for monitoring cache performance during intensive calendar scanning.
        """
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'utilization': len(self._cache) / self.maxsize,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                **self._stats
            }
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage of cache.
        
        This is approximate and depends on Python's object overhead.
        """
        import sys
        
        with self._lock:
            total_size = 0
            entry_count = len(self._cache)
            
            if entry_count == 0:
                return {'total_bytes': 0, 'avg_entry_bytes': 0, 'entry_count': 0}
            
            # Sample a few entries to estimate average size
            sample_keys = list(self._cache.keys())[:min(100, entry_count)]
            sample_size = 0
            
            for key in sample_keys:
                entry = self._cache[key]
                # Rough estimate of entry memory usage
                key_size = sys.getsizeof(key)
                value_size = sys.getsizeof(entry.value)
                entry_overhead = sys.getsizeof(entry)
                
                sample_size += key_size + value_size + entry_overhead
            
            avg_entry_size = sample_size / len(sample_keys)
            estimated_total = avg_entry_size * entry_count
            
            return {
                'total_bytes': estimated_total,
                'total_mb': estimated_total / (1024 * 1024),
                'avg_entry_bytes': avg_entry_size,
                'entry_count': entry_count
            }
    
    def keys(self) -> List[str]:
        """Get list of all keys in cache"""
        with self._lock:
            return list(self._cache.keys())
    
    def items(self) -> List[Tuple[str, Any]]:
        """Get list of all (key, value) pairs in cache"""
        with self._lock:
            return [(key, entry.value) for key, entry in self._cache.items()]
    
    def __len__(self) -> int:
        """Return current cache size"""
        return self.size()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired:
                self._remove_entry(key, 'expired')
                return False
            
            return True
    
    def __del__(self):
        """Cleanup when cache is destroyed"""
        self.stop()


class MultiTTLCache:
    """
    Multiple TTL caches with different configurations for different data types.
    
    Perfect for calendar spread calculators that need different caching strategies
    for prices (short TTL), historical data (long TTL), and analysis results.
    """
    
    def __init__(self):
        self.caches = {
            'price': TTLCache(maxsize=5000, default_ttl=60),      # 1 minute for prices
            'historical': TTLCache(maxsize=1000, default_ttl=3600), # 1 hour for historical
            'options': TTLCache(maxsize=2000, default_ttl=300),   # 5 minutes for options
            'analysis': TTLCache(maxsize=500, default_ttl=1800),  # 30 minutes for analysis
            'earnings': TTLCache(maxsize=1000, default_ttl=86400) # 24 hours for earnings
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get value from specific cache type"""
        if cache_type not in self.caches:
            self.logger.warning(f"Unknown cache type: {cache_type}")
            return None
        
        return self.caches[cache_type].get(key)
    
    def set(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in specific cache type"""
        if cache_type not in self.caches:
            self.logger.warning(f"Unknown cache type: {cache_type}")
            return
        
        self.caches[cache_type].set(key, value, ttl)
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches"""
        stats = {}
        total_size = 0
        total_hits = 0
        total_misses = 0
        
        for cache_type, cache in self.caches.items():
            cache_stats = cache.get_stats()
            stats[cache_type] = cache_stats
            total_size += cache_stats['size']
            total_hits += cache_stats['hits']
            total_misses += cache_stats['misses']
        
        total_requests = total_hits + total_misses
        combined_hit_rate = (total_hits / total_requests) if total_requests > 0 else 0.0
        
        stats['combined'] = {
            'total_size': total_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'combined_hit_rate': combined_hit_rate
        }
        
        return stats
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup all cache types"""
        results = {}
        for cache_type, cache in self.caches.items():
            results[cache_type] = cache.cleanup_expired()
        return results
    
    def stop_all(self) -> None:
        """Stop all caches"""
        for cache in self.caches.values():
            cache.stop()