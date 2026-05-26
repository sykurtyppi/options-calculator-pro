import React from 'react'

/**
 * Form for tuning alert thresholds. Each control writes a patch back
 * through `setConfig` (provided by useAlertConfig). Stateless — all
 * persistence happens in the hook.
 */
export default function AlertConfigPanel({ config, setConfig }) {
  return (
    <div className="alert-config-panel">
      <div className="oos-controls">
        <label className="oos-field">
          <span>Alerts Enabled</span>
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) => setConfig({ enabled: e.target.checked })}
          />
        </label>
        <label className="oos-field">
          <span>Min Setup Score</span>
          <input
            type="number"
            min="0"
            max="100"
            step="1"
            value={config.min_confidence}
            onChange={(e) => setConfig({ min_confidence: Number(e.target.value) })}
          />
        </label>
        <label className="oos-field">
          <span>Min Net Edge %</span>
          <input
            type="number"
            min="0"
            step="0.1"
            value={config.min_edge}
            onChange={(e) => setConfig({ min_edge: Number(e.target.value) })}
          />
        </label>
        <label className="oos-field">
          <span>Vol Regime Filter</span>
          <select
            value={config.vol_regime_filter}
            onChange={(e) => setConfig({ vol_regime_filter: e.target.value })}
          >
            <option value="any">Any</option>
            <option value="Low">Low</option>
            <option value="Normal">Normal</option>
            <option value="Elevated">Elevated</option>
            <option value="High">High</option>
          </select>
        </label>
      </div>
    </div>
  )
}
