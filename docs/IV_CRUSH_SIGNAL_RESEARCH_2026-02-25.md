# IV Crush / Calendar Spread Signal Research (2026-02-25)

## Primary sources reviewed
- Pricing Event Risk in Implied Volatility Curves (Review of Finance, 2025):  
  https://academic.oup.com/rof/advance-article/doi/10.1093/rof/rfaf038/8290883
- Option Pricing around Earnings Announcements (Review of Financial Studies, 2019):  
  https://academic.oup.com/rfs/article/32/12/4831/5305836
- Earnings Announcements and Equity Options (NBER, conference paper PDF):  
  https://conference.nber.org/confer/2005/MWs05/srivastava.pdf
- Pricing of Earnings Volatility and Predictability of Straddle Returns (Risks, 2023):  
  https://www.mdpi.com/2227-9091/11/1/11
- Cboe DataShop (Implied Earnings Moves dataset reference):  
  https://datashop.cboe.com/implied-earnings-moves
- The Stock-Earnings Effect in Option Implied Volatility (Cboe, 2024):  
  https://www.cboe.com/insights/posts/the-stock-earnings-effect-in-option-implied-volatility/
- Does Net Buying Pressure Affect the Shape of Implied Volatility Functions? (Journal of Finance, 2004):  
  https://ideas.repec.org/a/bla/jfinan/v59y2004i2p711-753.html
- What’s in the Spread? The Information Content of Option Bid-Ask Spreads (NBER Working Paper 9826):  
  https://www.nber.org/papers/w9826

## What matters for our system
- Event risk is a distinct component of the IV curve around earnings; generic non-event vol logic is not enough.
- Announcement timing and event-date alignment with short expiry materially affects realized behavior.
- Execution frictions (especially spread/quote quality) can erase weak apparent edge.
- Recent earnings moves are more informative than a single long-window aggregate; robust estimators are preferred.
- Smile concavity can indicate richer event-risk pricing, but also higher jump risk that must be charged in risk terms.
- Relative front-vs-back month IV richness is the direct event-premium check for earnings calendars.
- Estimated edge should be penalized by anchor uncertainty; small-sample anchors are prone to overstatement.

## Changes implemented in code
- Enforced short-leg/event alignment hard gates in `web/api/edge_engine.py`:
  - earnings must be strictly before short-leg expiry
  - minimum short-leg DTE
  - missing near-term quote/implied move -> hard no-trade
- Added spread-quality hard gate:
  - no-trade when near-term ATM spread exceeds threshold.
- Added robust move anchor:
  - computes blended anchor using median earnings move and recent 4-event average.
  - applies 1st/99th winsorization to historical move samples before anchor construction.
- Added concavity jump-risk surcharge to drawdown risk:
  - smile concavity can increase premium richness, but also tail risk.
- Added near/back-month event-premium hard gate:
  - no-trade when near-term ATM IV is not sufficiently richer than back-month ATM IV.
- Added near-term liquidity hard gate:
  - no-trade when near-term ATM OI/volume aggregate is below minimum executable threshold.
- Added uncertainty-adjusted edge estimation:
  - computes move-anchor uncertainty from dispersion + effective sample size.
  - subtracts uncertainty penalty before net-edge scoring.
- Exposed new metrics in API/UI:
  - `implied_vs_anchor_ratio`
  - `near_back_iv_ratio`
  - `near_term_liquidity_proxy`
  - `earnings_move_anchor_pct`
  - `earnings_move_avg_last4_pct`
  - `earnings_move_std_pct`
  - `move_uncertainty_pct`
  - `confidence_adjusted_gross_edge_pct`
  - `uncertainty_penalty_pct`
  - `near_term_spread_pct`
  - `concavity_risk_surcharge_pct`

## Still missing for true institutional quality
- Reliable BMO/AMC release-timing feed (current free feeds are inconsistent for exact release session).
- Split-adjusted slippage model tied to option notional and spread regime.
- Cross-sectional event-risk prior (symbol -> sector/beta cluster prior) for low-sample tickers.
