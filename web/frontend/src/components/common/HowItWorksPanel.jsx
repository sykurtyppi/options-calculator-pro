import React, { useState } from 'react'

// Plain-language onboarding for a first-time visitor. Replaces the old header
// link that pointed at a raw, unrendered architecture.md (engineer-facing). This
// answers "what does this do for me and why trust it" in plain English, and
// doubles as the glossary for the acronyms used across the screener and metrics.
//
// Uses explicit open state + conditional render (NOT <details>) so the rendered
// DOM and the render tests agree — jsdom includes <details> children in
// textContent regardless of open state, which has bitten this project before.

const GLOSSARY = [
  ['DTE', 'Days to earnings — calendar days until the report.'],
  ['Rel', 'Release timing — BMO (before market open) or AMC (after market close).'],
  [
    'IV / RV',
    'Implied vol ÷ realized vol — how richly options are priced versus the stock’s recent actual movement. Below 1.0 means options look cheap.',
  ],
  ['ATM IV', 'At-the-money implied volatility — the move the market is pricing into front-month options.'],
  [
    'TS (term structure)',
    'Front-month IV ÷ back-month IV — the pre-earnings “kink.” Steeper means more volatility is concentrated in the event itself.',
  ],
  ['Med Move', 'Median absolute stock move on past earnings days.'],
  ['N', 'Sample size — how many past earnings events back the stats. More is more reliable.'],
  [
    'Setup Score',
    'A composite ranking of setup quality (0–1). Ordering context only — NOT a calibrated win rate.',
  ],
]

const STEPS = [
  [
    'Screen',
    'Rank upcoming earnings by how cheap or rich their options look versus how the stock has actually moved on past earnings.',
  ],
  [
    'Drill in',
    'Open one ticker for the full volatility picture — term structure, historical earnings moves, and a setup-quality score.',
  ],
  [
    'Check the receipts',
    'Every qualifying signal is paper-traded forward, and the real out-of-sample track record is shown right here — including the losers.',
  ],
]

export default function HowItWorksPanel({ apiBase }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="how-it-works">
      <button
        type="button"
        className="docs-link"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? 'Hide explainer' : 'Learn how this works'}
      </button>

      {open && (
        <div className="how-it-works-panel" role="region" aria-label="How this works">
          <p className="hiw-lead">
            This tool finds stocks where options look mispriced heading into earnings, then shows
            you whether that edge has actually held up out-of-sample — so you can judge it for
            yourself instead of trusting a backtest.
          </p>

          <ol className="hiw-steps">
            {STEPS.map(([title, body]) => (
              <li key={title}>
                <strong>{title}.</strong> {body}
              </li>
            ))}
          </ol>

          <div className="hiw-glossary">
            <div className="hiw-glossary-title">Glossary</div>
            <dl>
              {GLOSSARY.map(([term, def]) => (
                <div className="hiw-glossary-row" key={term}>
                  <dt>{term}</dt>
                  <dd>{def}</dd>
                </div>
              ))}
            </dl>
          </div>

          <p className="hiw-honesty">
            Scores are setup-quality rankings, not win probabilities. We show forward results even
            when they’re bad, because a backtest that hides its losses isn’t evidence. Research
            only — not financial advice.
          </p>

          {apiBase && (
            <a
              className="hiw-tech-link"
              href={`${apiBase}/product-docs/architecture.md`}
              target="_blank"
              rel="noreferrer"
            >
              Technical architecture (for engineers) →
            </a>
          )}
        </div>
      )}
    </div>
  )
}
