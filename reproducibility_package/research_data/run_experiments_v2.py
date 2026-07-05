
"""
QKG-UAV V2
Quantum-Inspired Knowledge-Guided Decision Support for Risk-Aware UAV Mission Planning

Run:
    python src/run_experiments_v2.py

This version creates harder contested environments and evaluates:
    - Dijkstra
    - A*
    - Risk-Aware A*
    - Genetic Search
    - QKG-QI proposed method

Author: Dr. Sai Krishna Thota research support package
"""

from pathlib import Path
import heapq
import random
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# Reproducibility and folders
# -----------------------------

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "dataset"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

for p in [DATASET, RESULTS, FIGURES]:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Experiment settings
# -----------------------------

GRID = 35
N_SCENARIOS = 180

# We make the environment challenging enough that distance-only planning is not enough.
N_THREAT_CENTERS = 6
N_COMM_SHADOWS = 5
N_NO_FLY_PATCHES = 5


# -----------------------------
# Scenario generation
# -----------------------------

def gaussian_blob(grid, cx, cy, sigma):
    xg, yg = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))


def generate_scenario(sid, grid=GRID):
    """
    Generates a contested UAV mission map with:
        - threat intensity map
        - communication loss map
        - no-fly zones
        - weather penalty map
        - start and goal locations

    All maps are synthetic but deterministic for reproducibility.
    """
    rng = np.random.default_rng(SEED + sid)

    threat = rng.beta(2.2, 5.0, size=(grid, grid)) * 0.35
    comm_loss = rng.beta(2.0, 5.5, size=(grid, grid)) * 0.30
    weather = rng.beta(2.0, 7.0, size=(grid, grid)) * 0.20
    no_fly = np.zeros((grid, grid), dtype=int)

    # Threat clusters
    for _ in range(N_THREAT_CENTERS):
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(2.0, 5.2)
        amp = rng.uniform(0.45, 0.95)
        threat += amp * gaussian_blob(grid, cx, cy, sigma)

    # Communication shadow zones
    for _ in range(N_COMM_SHADOWS):
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(2.5, 6.0)
        amp = rng.uniform(0.35, 0.85)
        comm_loss += amp * gaussian_blob(grid, cx, cy, sigma)

    # Weather / turbulence corridors
    for _ in range(3):
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.20, 0.50)
        weather += amp * gaussian_blob(grid, cx, cy, sigma)

    threat = np.clip(threat, 0, 1)
    comm_loss = np.clip(comm_loss, 0, 1)
    weather = np.clip(weather, 0, 1)

    # No-fly zones, usually inside or near higher threat regions
    for _ in range(N_NO_FLY_PATCHES):
        cx, cy = rng.integers(4, grid - 4, size=2)
        radius = rng.integers(1, 4)
        for x in range(grid):
            for y in range(grid):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    no_fly[x, y] = 1

    start = (0, int(rng.integers(0, grid // 4)))
    goal = (grid - 1, int(rng.integers(3 * grid // 4, grid)))

    no_fly[start] = 0
    no_fly[goal] = 0
    threat[start] *= 0.2
    threat[goal] *= 0.2
    comm_loss[start] *= 0.2
    comm_loss[goal] *= 0.2

    # Energy budget differs by mission difficulty
    energy_budget = int(rng.integers(65, 95))
    risk_threshold = float(rng.uniform(0.25, 0.42))
    comm_threshold = float(rng.uniform(0.25, 0.45))

    return {
        "scenario_id": sid,
        "threat": threat,
        "comm_loss": comm_loss,
        "weather": weather,
        "no_fly": no_fly,
        "start": start,
        "goal": goal,
        "energy_budget": energy_budget,
        "risk_threshold": risk_threshold,
        "comm_threshold": comm_threshold,
    }


# -----------------------------
# Core path utilities
# -----------------------------

def neighbors(node, grid=GRID):
    x, y = node
    # 8-direction movement gives more realistic UAV route options.
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid and 0 <= ny < grid:
            yield (nx, ny), math.sqrt(dx * dx + dy * dy)


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def edge_cost(node, step_distance, scenario, mode):
    """
    mode:
        distance: mostly distance only
        risk: risk-aware A*
        knowledge: proposed method with knowledge-rule amplification
    """
    x, y = node
    if scenario["no_fly"][x, y] == 1:
        return np.inf

    threat = scenario["threat"][x, y]
    comm = scenario["comm_loss"][x, y]
    weather = scenario["weather"][x, y]

    if mode == "distance":
        return step_distance

    if mode == "risk":
        return step_distance + 2.5 * threat + 1.4 * comm + 0.8 * weather

    # Knowledge-guided cost
    cost = step_distance + 4.5 * threat + 2.8 * comm + 1.2 * weather

    # Knowledge rules: these represent mission-domain constraints.
    if threat > 0.55:
        cost += 7.0
    if threat > 0.75:
        cost += 15.0
    if comm > 0.50:
        cost += 4.0
    if weather > 0.55:
        cost += 2.0

    # Soft corridor preference: keep communication loss acceptable.
    if comm <= scenario["comm_threshold"] and threat <= scenario["risk_threshold"]:
        cost -= 0.35

    return max(cost, 0.01)


def reconstruct(came, goal):
    if goal not in came:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    return path[::-1]


def graph_search(scenario, astar=True, mode="distance"):
    start, goal = scenario["start"], scenario["goal"]
    pq = [(0.0, start)]
    came = {start: None}
    cost = {start: 0.0}
    expanded = 0

    while pq:
        _, current = heapq.heappop(pq)
        expanded += 1

        if current == goal:
            break

        for nb, dist in neighbors(current, scenario["threat"].shape[0]):
            ec = edge_cost(nb, dist, scenario, mode)
            if np.isinf(ec):
                continue

            nc = cost[current] + ec
            if nb not in cost or nc < cost[nb]:
                cost[nb] = nc
                h = euclidean(nb, goal) if astar else 0.0
                came[nb] = current
                heapq.heappush(pq, (nc + h, nb))

    return reconstruct(came, goal), expanded


# -----------------------------
# Metrics
# -----------------------------

def path_metrics(path, scenario):
    if not path:
        return {
            "success": 0,
            "avg_path_risk": np.nan,
            "max_path_risk": np.nan,
            "energy_cost": np.nan,
            "avg_comm_loss": np.nan,
            "avg_weather_penalty": np.nan,
            "constraint_violations": 999,
            "knowledge_rule_compliance": 0.0,
            "path_length": 0,
            "mission_objective_score": np.nan
        }

    threat_vals = np.array([scenario["threat"][x, y] for x, y in path])
    comm_vals = np.array([scenario["comm_loss"][x, y] for x, y in path])
    weather_vals = np.array([scenario["weather"][x, y] for x, y in path])
    no_fly_vals = np.array([scenario["no_fly"][x, y] for x, y in path])

    # Energy includes distance-like path length and environmental penalties.
    energy = len(path) + 0.35 * threat_vals.sum() + 0.25 * comm_vals.sum() + 0.20 * weather_vals.sum()

    risk_viol = np.sum(threat_vals > scenario["risk_threshold"])
    comm_viol = np.sum(comm_vals > scenario["comm_threshold"])
    hard_viol = np.sum(no_fly_vals > 0)

    violations = int(risk_viol + comm_viol + 10 * hard_viol)

    compliance = 1.0 - min(1.0, violations / max(1, 2 * len(path)))

    success = int(
        path[-1] == scenario["goal"]
        and hard_viol == 0
        and energy <= scenario["energy_budget"] * 1.25
        and np.mean(threat_vals) <= 0.65
    )

    objective = (
        energy
        + 60.0 * float(np.mean(threat_vals))
        + 35.0 * float(np.mean(comm_vals))
        + 20.0 * float(np.mean(weather_vals))
        + 8.0 * violations
    )

    return {
        "success": success,
        "avg_path_risk": float(np.mean(threat_vals)),
        "max_path_risk": float(np.max(threat_vals)),
        "energy_cost": float(energy),
        "avg_comm_loss": float(np.mean(comm_vals)),
        "avg_weather_penalty": float(np.mean(weather_vals)),
        "constraint_violations": violations,
        "knowledge_rule_compliance": float(compliance),
        "path_length": len(path),
        "mission_objective_score": float(objective)
    }


# -----------------------------
# Genetic and QKG-QI methods
# -----------------------------

def greedy_random_path(scenario, mode, noise=4.0, max_steps=None):
    if max_steps is None:
        max_steps = GRID * 3

    start, goal = scenario["start"], scenario["goal"]
    cur = start
    path = [cur]

    for _ in range(max_steps):
        if cur == goal:
            break

        candidates = []
        for nb, dist in neighbors(cur, GRID):
            if scenario["no_fly"][nb] == 1:
                continue
            score = (
                euclidean(nb, goal)
                + edge_cost(nb, dist, scenario, mode)
                + random.random() * noise
            )
            candidates.append((score, nb))

        if not candidates:
            return []

        candidates.sort(key=lambda z: z[0])
        # Sometimes choose second or third best to imitate genetic exploration.
        pick = 0
        if len(candidates) > 2 and random.random() < 0.25:
            pick = random.randint(1, min(2, len(candidates) - 1))
        cur = candidates[pick][1]
        path.append(cur)

        # Prevent loops from getting too long.
        if len(path) > max_steps:
            return []

    return path if path and path[-1] == goal else []


def genetic_search(scenario, trials=55):
    best_path = []
    best_score = np.inf

    for _ in range(trials):
        path = greedy_random_path(scenario, mode="risk", noise=5.5, max_steps=GRID * 3)
        m = path_metrics(path, scenario)
        score = m["mission_objective_score"]
        if not np.isnan(score) and score < best_score:
            best_score = score
            best_path = path

    return best_path, trials


def qkg_qi_search(scenario):
    """
    Proposed QKG-QI method.

    This is a practical quantum-inspired optimizer:
    - knowledge-aware weighted graph search gives a strong feasible route
    - annealed candidate exploration imitates probability-amplitude-style exploration
    - objective combines mission knowledge, risk, communication loss, weather, and energy

    We do not claim quantum hardware execution.
    """
    seed_path, expanded = graph_search(scenario, astar=True, mode="knowledge")
    candidates = []

    if seed_path:
        candidates.append(seed_path)

    # Annealed randomized exploration around knowledge-guided routing.
    temperatures = np.linspace(6.0, 0.5, 65)
    for t in temperatures:
        path = greedy_random_path(
            scenario,
            mode="knowledge",
            noise=float(t),
            max_steps=GRID * 3
        )
        if path:
            candidates.append(path)

    if not candidates:
        return [], expanded + len(temperatures)

    # Quantum-inspired measurement step: select lowest mission objective.
    best = min(candidates, key=lambda p: path_metrics(p, scenario)["mission_objective_score"])
    return best, expanded + len(temperatures)


# -----------------------------
# Ablation variants
# -----------------------------

def ablation_variant(scenario, variant):
    if variant == "Full QKG-QI":
        return qkg_qi_search(scenario)[0]

    if variant == "No knowledge rules":
        return graph_search(scenario, astar=True, mode="risk")[0]

    if variant == "No communication rule":
        modified = dict(scenario)
        modified["comm_threshold"] = 1.0
        return qkg_qi_search(modified)[0]

    if variant == "No risk amplification":
        modified = dict(scenario)
        modified["risk_threshold"] = 1.0
        return qkg_qi_search(modified)[0]

    return []


# -----------------------------
# Figures
# -----------------------------

def save_representative_path_figure(scenario):
    methods_paths = {
        "A*": graph_search(scenario, astar=True, mode="distance")[0],
        "Risk-Aware A*": graph_search(scenario, astar=True, mode="risk")[0],
        "GA": genetic_search(scenario, trials=55)[0],
        "QKG-QI": qkg_qi_search(scenario)[0],
    }

    plt.figure(figsize=(8, 7))
    plt.imshow(scenario["threat"].T, origin="lower")
    nf = np.argwhere(scenario["no_fly"] == 1)
    if len(nf) > 0:
        plt.scatter(nf[:, 0], nf[:, 1], marker="s", s=18, label="No-fly cells")

    for name, path in methods_paths.items():
        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            plt.plot(xs, ys, linewidth=2, label=name)

    start, goal = scenario["start"], scenario["goal"]
    plt.scatter([start[0]], [start[1]], marker="o", s=100, label="Start")
    plt.scatter([goal[0]], [goal[1]], marker="x", s=120, label="Goal")
    plt.title("Contested UAV Risk Heatmap and Route Comparison")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "v2_risk_heatmap_route_comparison.png", dpi=300)
    plt.close()


def save_summary_figures(summary, ablation):
    order = ["Dijkstra", "A*", "Risk-Aware A*", "GA", "QKG-QI"]
    s = summary.set_index("method").loc[order]

    figures = [
        ("avg_path_risk", "Average Path Risk by Method", "Average path risk", "v2_avg_path_risk.png"),
        ("avg_comm_loss", "Average Communication Loss by Method", "Average communication loss", "v2_avg_comm_loss.png"),
        ("avg_energy_cost", "Average Energy Cost by Method", "Energy cost", "v2_avg_energy_cost.png"),
        ("avg_constraint_violations", "Average Constraint Violations by Method", "Constraint violations", "v2_constraint_violations.png"),
        ("avg_rule_compliance", "Knowledge-Rule Compliance by Method", "Compliance", "v2_rule_compliance.png"),
        ("avg_objective_score", "Mission Objective Score by Method", "Objective score", "v2_objective_score.png"),
    ]

    for col, title, ylabel, fname in figures:
        plt.figure(figsize=(8, 5))
        s[col].plot(kind="bar")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES / fname, dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    ablation.set_index("variant")["mean_objective_score"].plot(kind="bar")
    plt.title("QKG-QI Ablation Study")
    plt.ylabel("Mean objective score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "v2_ablation_study.png", dpi=300)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def run():
    scenario_rows = []
    result_rows = []

    methods = ["Dijkstra", "A*", "Risk-Aware A*", "GA", "QKG-QI"]

    for sid in range(N_SCENARIOS):
        scenario = generate_scenario(sid)

        scenario_rows.append({
            "scenario_id": sid,
            "grid_size": GRID,
            "start": str(scenario["start"]),
            "goal": str(scenario["goal"]),
            "energy_budget": scenario["energy_budget"],
            "risk_threshold": scenario["risk_threshold"],
            "comm_threshold": scenario["comm_threshold"],
            "mean_threat": float(scenario["threat"].mean()),
            "max_threat": float(scenario["threat"].max()),
            "mean_comm_loss": float(scenario["comm_loss"].mean()),
            "mean_weather": float(scenario["weather"].mean()),
            "no_fly_cells": int(scenario["no_fly"].sum()),
        })

        for method in methods:
            t0 = time.perf_counter()

            if method == "Dijkstra":
                path, steps = graph_search(scenario, astar=False, mode="distance")
            elif method == "A*":
                path, steps = graph_search(scenario, astar=True, mode="distance")
            elif method == "Risk-Aware A*":
                path, steps = graph_search(scenario, astar=True, mode="risk")
            elif method == "GA":
                path, steps = genetic_search(scenario, trials=55)
            elif method == "QKG-QI":
                path, steps = qkg_qi_search(scenario)
            else:
                path, steps = [], 0

            runtime_ms = (time.perf_counter() - t0) * 1000.0
            m = path_metrics(path, scenario)

            result_rows.append({
                "scenario_id": sid,
                "method": method,
                "success": m["success"],
                "avg_path_risk": m["avg_path_risk"],
                "max_path_risk": m["max_path_risk"],
                "energy_cost": m["energy_cost"],
                "avg_comm_loss": m["avg_comm_loss"],
                "avg_weather_penalty": m["avg_weather_penalty"],
                "constraint_violations": m["constraint_violations"],
                "knowledge_rule_compliance": m["knowledge_rule_compliance"],
                "path_length": m["path_length"],
                "mission_objective_score": m["mission_objective_score"],
                "expanded_or_iterations": steps,
                "runtime_ms": runtime_ms,
            })

    scenarios_df = pd.DataFrame(scenario_rows)
    results_df = pd.DataFrame(result_rows)

    scenarios_df.to_csv(DATASET / "uav_mission_scenarios_v2.csv", index=False)
    results_df.to_csv(RESULTS / "baseline_comparison_v2.csv", index=False)

    summary = results_df.groupby("method").agg(
        success_rate=("success", "mean"),
        avg_path_risk=("avg_path_risk", "mean"),
        std_path_risk=("avg_path_risk", "std"),
        avg_max_path_risk=("max_path_risk", "mean"),
        avg_energy_cost=("energy_cost", "mean"),
        avg_comm_loss=("avg_comm_loss", "mean"),
        avg_weather_penalty=("avg_weather_penalty", "mean"),
        avg_constraint_violations=("constraint_violations", "mean"),
        avg_rule_compliance=("knowledge_rule_compliance", "mean"),
        avg_objective_score=("mission_objective_score", "mean"),
        avg_runtime_ms=("runtime_ms", "mean"),
        avg_path_length=("path_length", "mean"),
    ).reset_index()

    summary.to_csv(RESULTS / "statistical_summary_v2.csv", index=False)

    # Significance tests: proposed method vs all baselines
    tests = []
    qkg = results_df[results_df["method"] == "QKG-QI"].sort_values("scenario_id")

    for baseline in ["Dijkstra", "A*", "Risk-Aware A*", "GA"]:
        b = results_df[results_df["method"] == baseline].sort_values("scenario_id")
        for metric in [
            "avg_path_risk",
            "energy_cost",
            "avg_comm_loss",
            "constraint_violations",
            "mission_objective_score"
        ]:
            x = b[metric].astype(float).fillna(b[metric].mean())
            y = qkg[metric].astype(float).fillna(qkg[metric].mean())
            t_stat, p_val = stats.ttest_rel(x, y)
            tests.append({
                "comparison": f"QKG-QI vs {baseline}",
                "metric": metric,
                "baseline_mean": float(x.mean()),
                "qkg_qi_mean": float(y.mean()),
                "paired_t_statistic": float(t_stat),
                "p_value": float(p_val),
            })

    pd.DataFrame(tests).to_csv(RESULTS / "significance_tests_v2.csv", index=False)

    # Ablation on the SAME 180 scenarios used for the baseline comparison
    # (V2 fix: earlier version used a different 45-scenario seed block,
    # which made Table 1 and Table 3 QKG-QI means non-comparable).
    ablation_rows = []
    variants = [
        "Full QKG-QI",
        "No knowledge rules",
        "No communication rule",
        "No risk amplification",
    ]

    for variant in variants:
        metrics_list = []
        for sid in range(N_SCENARIOS):
            scenario = generate_scenario(sid)
            path = ablation_variant(scenario, variant)
            metrics_list.append(path_metrics(path, scenario))

        ablation_rows.append({
            "variant": variant,
            "mean_success_rate": float(np.mean([m["success"] for m in metrics_list])),
            "mean_path_risk": float(np.nanmean([m["avg_path_risk"] for m in metrics_list])),
            "mean_comm_loss": float(np.nanmean([m["avg_comm_loss"] for m in metrics_list])),
            "mean_constraint_violations": float(np.mean([m["constraint_violations"] for m in metrics_list])),
            "mean_rule_compliance": float(np.mean([m["knowledge_rule_compliance"] for m in metrics_list])),
            "mean_objective_score": float(np.nanmean([m["mission_objective_score"] for m in metrics_list])),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(RESULTS / "ablation_study_v2.csv", index=False)

    # Figures
    save_representative_path_figure(generate_scenario(12))
    save_summary_figures(summary, ablation_df)

    print("\n=== QKG-UAV V2 experiment completed ===\n")
    print(summary.to_string(index=False))
    print("\nFiles saved in:")
    print(f"  {RESULTS}")
    print(f"  {FIGURES}")
    print(f"  {DATASET}")


if __name__ == "__main__":
    run()
