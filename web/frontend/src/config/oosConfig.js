export const TODAY_ISO = new Date().toISOString().split('T')[0]

export const DEFAULT_OOS_PARAMS = {
  oos_stability_profile: 'stability_auto',
  lookback_days: '1095',
  max_backtest_symbols: '50',
  backtest_start_date: '2023-01-01',
  backtest_end_date: TODAY_ISO,
  min_signal_score: '0.50',
  min_crush_confidence: '0.30',
  min_crush_magnitude: '0.06',
  min_crush_edge: '0.025',
  target_entry_dte: '6',
  entry_dte_band: '5',
  min_daily_share_volume: '1500000',
  max_abs_momentum_5d: '0.11',
  oos_train_days: '189',
  oos_test_days: '42',
  oos_step_days: '42',
  oos_top_n_train: '1',
  oos_min_splits: '8',
  oos_min_total_test_trades: '80',
  oos_min_trades_per_split: '5.0',
}

export const OOS_PROFILE_PRESETS = {
  stability_auto: {
    min_signal_score: '0.50', min_crush_confidence: '0.30', min_crush_magnitude: '0.06',
    min_crush_edge: '0.025', min_daily_share_volume: '1500000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '5',
  },
  evidence_balanced: {
    min_signal_score: '0.48', min_crush_confidence: '0.28', min_crush_magnitude: '0.06',
    min_crush_edge: '0.025', min_daily_share_volume: '1500000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '5',
  },
  sample_expansion: {
    min_signal_score: '0.45', min_crush_confidence: '0.25', min_crush_magnitude: '0.05',
    min_crush_edge: '0.015', min_daily_share_volume: '1000000', max_abs_momentum_5d: '0.11',
    target_entry_dte: '6', entry_dte_band: '6',
  },
  variance_control: {
    min_signal_score: '0.65', min_crush_confidence: '0.50', min_crush_magnitude: '0.09',
    min_crush_edge: '0.025', min_daily_share_volume: '10000000', max_abs_momentum_5d: '0.09',
    target_entry_dte: '6', entry_dte_band: '4',
  },
  alpha_focus: {
    min_signal_score: '0.65', min_crush_confidence: '0.50', min_crush_magnitude: '0.09',
    min_crush_edge: '0.03', min_daily_share_volume: '5000000', max_abs_momentum_5d: '0.08',
    target_entry_dte: '6', entry_dte_band: '3',
  },
}

export const OOS_PROFILE_LABELS = {
  stability_auto: 'Auto',
  evidence_balanced: 'Evidence Balanced',
  sample_expansion: 'Sample Expansion',
  variance_control: 'Variance Control',
  alpha_focus: 'Alpha Focus',
}
