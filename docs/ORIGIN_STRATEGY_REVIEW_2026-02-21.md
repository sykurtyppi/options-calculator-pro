# Origin Strategy Review (Recovered from PDF)

Date: 2026-02-21  
Source reviewed: `/Users/tristanalejandro/Downloads/option strategy volitility .pdf`

## What the original version focused on

- Entry/exit timing around earnings:
  - Open roughly 15 minutes before earnings announcement close.
  - "Jump" target: close 15 minutes after the next open.
  - "Move" target: close 15 minutes before next market close.
- Calendar spread mechanics:
  - Short front-month, long back-month, same strike.
  - Emphasis on IV crush in short leg while retaining value in long leg.
- Core predictor concepts:
  - Term-structure slope.
  - Front/back IV ratio.
  - Implied move richness vs historical move.
- Practical notes:
  - Include commissions/slippage in realized PnL.
  - Strategy examples used earnings names like FDX/NKE and date windows.

## Technical origin

- Early implementation appears to have started as `earnings_checker.py` in a `trading_ml_project`.
- UI and logic traces in the PDF indicate an earlier `FreeSimpleGUI` workflow that later evolved into current PySide architecture.

## Institutional interpretation

- The underlying alpha thesis is valid:
  - monetize event-vol premium and term-structure dislocation with controlled directional exposure.
- The strategy quality hinges on:
  - expiry pair selection,
  - debit efficiency vs implied move,
  - liquidity/spread quality,
  - event-timing alignment.

## Improvements integrated in current codebase (this pass)

- Better expiry-pair selection logic for calendar setups.
- Calendar-specific max-profit and break-even heuristics tied to:
  - IV premium,
  - theta edge,
  - expected move,
  - moneyness penalty.
- Improved edge scoring with new components:
  - term-structure fit,
  - debit efficiency.
- ML feature prep enriched with:
  - time-decay ratio,
  - liquidity score,
  - volatility skew/event premium.
- Probability-of-profit display normalization fixed (decimal -> true percent display).

## Additional upgrades (second pass)

- Event-aligned expiry selection:
  - short leg is now preferentially chosen at/just after earnings timing when possible.
  - long leg is chosen with a controlled tenor gap (roughly 2-10 weeks) for cleaner calendar structure.
- Execution realism integrated:
  - estimated transaction/slippage cost per contract is computed from spreads and liquidity.
  - edge scoring includes execution quality.
  - risk metrics now use effective debit (including transaction costs), affecting max loss, max profit, breakeven width, and expected value.

## Remaining high-impact next steps

- Add event-specific historical option IV panel data for true IV crush calibration.
- Replace heuristic payoff estimates with scenario pricing of both legs through the holding window.
- Build out-of-sample walk-forward validation by earnings cohort and liquidity bucket.
- Add transaction cost model by spread width, ADV, and time-of-day.
