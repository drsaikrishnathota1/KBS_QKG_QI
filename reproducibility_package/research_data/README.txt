Research Data Package
Manuscript: KNOSYS-D-26-09565
"Quantum-Inspired Knowledge-Guided Decision Support for Risk-Aware UAV Mission
Planning in Contested Environments"
Author: Sai Krishna Thota

CONTENTS

1. Source code (Python 3)
   - run_experiments_v2.py
       Main experiment script. Generates the 180-scenario synthetic benchmark,
       runs all five methods (Dijkstra, A*, Risk-Aware A*, GA, QKG-QI),
       and produces the baseline comparison (Table 1), significance tests
       (Table 2 / Supplementary Table S1), and ablation study (Table 3).
       Uses fixed random seeds throughout for full reproducibility.
   - sensitivity_sweep.py
       Sensitivity analysis on the threat-weight coefficient of the
       knowledge-guided edge cost. Produces sensitivity_threat_weight.csv
       (Supplementary Table S3).
   - stress_test.py
       Stress-test script using a harder scenario configuration (denser
       threats, denser no-fly patches, tighter energy budget) to
       characterize QKG-QI's failure mode. Referenced in Section 5
       (Limitations) and Supplementary Material S6.
   - reviewer3_metrics.py
       Computes (a) paired Cohen's d effect sizes with Bonferroni
       correction across all baseline comparisons, producing
       effect_sizes_v2.csv (Supplementary Table S2), and (b) the
       rule-attribution coverage metric across all 180 QKG-QI routes,
       producing rule_attribution_coverage_v2.csv (Supplementary S9).
   - regen_figure1.py
       Generates Figure 1 (route-comparison visualization) using the
       real simulation output, with the improved styling requested by
       Reviewers 1 and 4 (colorbar, per-method line styles, legend,
       high-contrast start/goal markers).

2. Result data (CSV)
   - ablation_study_v2_corrected.csv
       Corrected ablation study results (Table 3), rerun on the
       identical 180-scenario set used in the baseline comparison
       (Table 1), resolving the scenario-mismatch inconsistency
       independently flagged by Reviewers 1 and 2.
   - effect_sizes_v2.csv
       Full effect-size table for all 20 baseline comparisons
       (4 baselines x 5 metrics), referenced in response to
       Reviewer 3's comment on statistical significance and
       multiple-comparisons correction.
   - sensitivity_threat_weight.csv
       Full sensitivity sweep results referenced in response to
       Reviewer 1's comment on the risk-energy trade-off.
   - rule_attribution_coverage_v2.csv
       Per-route rule-attribution coverage values (fraction of added
       routing cost attributable to named knowledge rules) for all
       180 QKG-QI routes in the main benchmark, referenced in response
       to Reviewers 2 and 3's comments on explainability.

REPRODUCIBILITY NOTES

All scripts use fixed random seeds (see the SEED constant and per-scenario
seed offsets in run_experiments_v2.py). Running run_experiments_v2.py from
a clean environment reproduces Tables 1-3 exactly. Python dependencies:
numpy, pandas, scipy, matplotlib. No external or proprietary data sources
are used; all scenarios are synthetically generated as described in
Section 3 of the manuscript.

CONTACT

Sai Krishna Thota
drsaikrishnathota@ieee.org
ORCID: 0009-0008-5246-9421
