"""
Sensitivity sweep for QKG-QI: vary the threat-weight coefficient (currently
hardcoded 4.5 in edge_cost) and report the risk/energy/objective trade-off.
Runs on a 40-scenario subset (sid 0..39) for runtime reasons; same generator
and seeds as the main experiment, so results are directly comparable.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from run_experiments_v2 import generate_scenario, path_metrics, greedy_random_path, graph_search
import run_experiments_v2 as m

N = 40
weights = [2.0, 3.25, 4.5, 6.0, 8.0]  # 4.5 = original manuscript value

rows = []
for w in weights:
    # monkey-patch the threat coefficient used inside edge_cost's "knowledge" branch
    orig_edge_cost = m.edge_cost

    def patched_edge_cost(node, step_distance, scenario, mode, w=w, orig=orig_edge_cost):
        x, y = node
        if scenario["no_fly"][x, y] == 1:
            return np.inf
        threat = scenario["threat"][x, y]
        comm = scenario["comm_loss"][x, y]
        weather = scenario["weather"][x, y]
        if mode != "knowledge":
            return orig(node, step_distance, scenario, mode)
        cost = step_distance + w * threat + 2.8 * comm + 1.2 * weather
        if threat > 0.55:
            cost += 7.0
        if threat > 0.75:
            cost += 15.0
        if comm > 0.50:
            cost += 4.0
        if weather > 0.55:
            cost += 2.0
        if comm <= scenario["comm_threshold"] and threat <= scenario["risk_threshold"]:
            cost -= 0.35
        return max(cost, 0.01)

    m.edge_cost = patched_edge_cost

    metrics_list = []
    for sid in range(N):
        scenario = generate_scenario(sid)
        seed_path, _ = graph_search(scenario, astar=True, mode="knowledge")
        candidates = [seed_path] if seed_path else []
        temperatures = np.linspace(6.0, 0.5, 65)
        for t in temperatures:
            p = greedy_random_path(scenario, mode="knowledge", noise=float(t), max_steps=m.GRID * 3)
            if p:
                candidates.append(p)
        if not candidates:
            continue
        best = min(candidates, key=lambda p: path_metrics(p, scenario)["mission_objective_score"])
        metrics_list.append(path_metrics(best, scenario))

    m.edge_cost = orig_edge_cost

    rows.append({
        "threat_weight": w,
        "mean_risk": float(np.mean([x["avg_path_risk"] for x in metrics_list])),
        "mean_energy": float(np.mean([x["energy_cost"] for x in metrics_list])),
        "mean_objective": float(np.mean([x["mission_objective_score"] for x in metrics_list])),
        "mean_violations": float(np.mean([x["constraint_violations"] for x in metrics_list])),
    })

df = pd.DataFrame(rows)
df.to_csv("results/sensitivity_threat_weight.csv", index=False)
print(df.to_string(index=False))
