import { useState } from 'react'

const LS_KEY = 'watchlist_v1'
const MAX_ENTRIES = 20

/**
 * Persistent watchlist hook backed by localStorage.
 *
 * Entries are de-duped by symbol (newer wins) and capped at 20 entries.
 * Quota errors are swallowed silently — the in-memory state stays correct
 * even if localStorage refuses the write (Safari private mode, etc.).
 */
export function useWatchlist() {
  const [watchlist, setWatchlist] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(LS_KEY) || '[]')
    } catch {
      return []
    }
  })

  function addToWatchlist(entry) {
    setWatchlist((prev) => {
      const filtered = prev.filter((e) => e.symbol !== entry.symbol)
      const next = [entry, ...filtered].slice(0, MAX_ENTRIES)
      try {
        localStorage.setItem(LS_KEY, JSON.stringify(next))
      } catch {
        /* quota or disabled storage — keep the in-memory copy */
      }
      return next
    })
  }

  function removeFromWatchlist(sym) {
    setWatchlist((prev) => {
      const next = prev.filter((e) => e.symbol !== sym)
      try {
        localStorage.setItem(LS_KEY, JSON.stringify(next))
      } catch {
        /* quota or disabled storage */
      }
      return next
    })
  }

  function isInWatchlist(sym) {
    return watchlist.some((e) => e.symbol === sym)
  }

  return { watchlist, addToWatchlist, removeFromWatchlist, isInWatchlist }
}
