import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------
# 1. Effect sizes (Cohen's d for paired samples) + Bonferroni correction
# ---------------------------------------------------------------
results_df = pd.read_csv("results/baseline_comparison_v2.csv")

qkg = results_df[results_df["method"] == "QKG-QI"].sort_values("scenario_id")
rows = []
metrics = ["avg_path_risk", "energy_cost", "avg_comm_loss", "constraint_violations", "mission_objective_score"]
baselines = ["Dijkstra", "A*", "Risk-Aware A*", "GA"]
n_tests = len(baselines) * len(metrics)
alpha_bonf = 0.05 / n_tests

for baseline in baselines:
    b = results_df[results_df["method"] == baseline].sort_values("scenario_id")
    for metric in metrics:
        x = b[metric].astype(float).fillna(b[metric].mean())
        y = qkg[metric].astype(float).fillna(qkg[metric].mean())
        diff = x.values - y.values
        d = diff.mean() / diff.std(ddof=1)  # paired Cohen's d
        t_stat, p_val = stats.ttest_rel(x, y)
        rows.append({
            "comparison": f"QKG-QI vs {baseline}",
            "metric": metric,
            "cohens_d": round(float(d), 3),
            "p_value": p_val,
            "significant_after_bonferroni": p_val < alpha_bonf,
        })

eff = pd.DataFrame(rows)
eff.to_csv("results/effect_sizes_v2.csv", index=False)
print(f"Number of comparisons: {n_tests}, Bonferroni alpha: {alpha_bonf:.2e}")
print(eff.to_string(index=False))
print()

# ---------------------------------------------------------------
# 2. Quantitative rule-attribution ("explainability coverage") metric
#    Fraction of QKG-QI's added routing cost (beyond raw distance) that
#    is attributable to explicit, named knowledge rules K(v) vs. the
#    continuous environmental terms (alpha*T + beta*C + gamma*W).
# ---------------------------------------------------------------
import sys
sys.path.insert(0, "src")
from run_experiments_v2 import generate_scenario, qkg_qi_search

def rule_attribution(node, scenario):
    x, y = node
    T = scenario["threat"][x, y]
    C = scenario["comm_loss"][x, y]
    W = scenario["weather"][x, y]
    continuous = 4.5 * T + 2.8 * C + 1.2 * W
    rule = 0.0
    if T > 0.55:
        rule += 7.0
    if T > 0.75:
        rule += 15.0
    if C > 0.50:
        rule += 4.0
    if W > 0.55:
        rule += 2.0
    if C <= scenario["comm_threshold"] and T <= scenario["risk_threshold"]:
        rule -= 0.35
    return continuous, rule

coverage_list = []
for sid in range(180):
    scenario = generate_scenario(sid)
    path, _ = qkg_qi_search(scenario)
    if not path:
        continue
    total_continuous = 0.0
    total_rule = 0.0
    for node in path:
        c, r = rule_attribution(node, scenario)
        total_continuous += c
        total_rule += abs(r)  # magnitude of rule-driven adjustment (positive or the corridor bonus)
    denom = total_continuous + total_rule
    if denom > 0:
        coverage_list.append(total_rule / denom)

coverage_arr = np.array(coverage_list)
print(f"Rule-attribution coverage across {len(coverage_arr)} routes:")
print(f"  mean = {coverage_arr.mean():.3f}, median = {np.median(coverage_arr):.3f}, "
      f"min = {coverage_arr.min():.3f}, max = {coverage_arr.max():.3f}")

pd.DataFrame({"rule_attribution_fraction": coverage_arr}).to_csv(
    "results/rule_attribution_coverage_v2.csv", index=False)
