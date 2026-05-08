# QKG-UAV V2: Quantum-Inspired Knowledge-Guided UAV Decision Support

This is Version 2 of the reproducible simulation package for a Knowledge-Based Systems Short Communication.

## Goal

Version 2 creates a harder contested UAV route-planning benchmark where the proposed **QKG-QI** method has clearer advantages in:
- risk reduction
- communication reliability
- knowledge-rule compliance
- mission objective score

## Run in Cursor / VS Code / PyCharm

```bash
cd qkg_uav_v2_cursor
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python src/run_experiments_v2.py
```

## Outputs

After running, check:

- `dataset/uav_mission_scenarios_v2.csv`
- `results/baseline_comparison_v2.csv`
- `results/statistical_summary_v2.csv`
- `results/significance_tests_v2.csv`
- `results/ablation_study_v2.csv`
- `figures/*.png`

## Important manuscript wording

Use **quantum-inspired** only. Do not claim real quantum hardware execution.

This package generates synthetic but reproducible benchmark data using fixed random seeds.
