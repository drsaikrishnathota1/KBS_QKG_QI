# Version 2 Manuscript Notes

## Why V2 is stronger

Version 2 creates a harder contested UAV mission environment with:
- larger grid size
- threat clusters
- communication shadow zones
- no-fly areas
- weather/turbulence penalty
- mission thresholds for risk and communication reliability

## Proposed Method Name

QKG-QI:
Quantum-Inspired Knowledge-Guided optimizer

## Baselines

1. Dijkstra
2. A*
3. Risk-Aware A*
4. Genetic Search
5. Proposed QKG-QI

## Metrics

- Mission success rate
- Average path risk
- Maximum path risk
- Energy cost
- Average communication loss
- Weather penalty
- Constraint violations
- Knowledge-rule compliance
- Mission objective score
- Runtime

## Suggested manuscript claim

The proposed method is designed to improve mission-level decision quality by integrating symbolic mission rules with quantum-inspired probabilistic search and risk-aware objective scoring.

## Important limitation to state

The benchmark is synthetic and reproducible; it is not a real UAV flight test. Future work should evaluate with hardware-in-loop, public UAV telemetry, or real-world geospatial threat layers.
