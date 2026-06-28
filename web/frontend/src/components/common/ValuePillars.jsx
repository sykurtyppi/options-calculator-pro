import React from 'react'

// Always-visible, scannable value strip under the header. The header tagline is
// the one-line hook; the collapsible HowItWorksPanel is the deep dive on demand;
// this sits between them so a first-time visitor grasps what the tool does — and
// why to trust it — in a few seconds without clicking anything. The middle
// pillar leads with the honesty wedge, which is the real differentiator.
const PILLARS = [
  [
    'Spot mispriced earnings options',
    'Rank upcoming reports by how rich or cheap their options look versus how the stock has actually moved on past earnings.',
  ],
  [
    'Proof, not promises',
    'Every qualifying signal is paper-traded forward and the real out-of-sample track record is shown right here — losers included.',
  ],
  [
    'A curated universe',
    'A focused set of liquid, earnings-driven names — signal over noise, not an uncapped scanner spraying low-quality setups.',
  ],
]

export default function ValuePillars() {
  return (
    <ul className="value-pillars" aria-label="What this tool does">
      {PILLARS.map(([title, body]) => (
        <li className="value-pillar" key={title}>
          {/* Not a heading element: these are value-prop blurbs, and making them
              <h2> would imply the real <h3> sections below are nested under the
              last pillar — a false outline for screen-reader heading nav. */}
          <div className="value-pillar-title">{title}</div>
          <p className="value-pillar-body">{body}</p>
        </li>
      ))}
    </ul>
  )
}
