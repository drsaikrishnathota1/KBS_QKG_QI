import sys
sys.path.insert(0, "src")
import numpy as np
from run_experiments_v2 import (
    generate_scenario, qkg_qi_search, path_metrics, GRID,
    N_THREAT_CENTERS, N_NO_FLY_PATCHES
)
import run_experiments_v2 as m

# Harder scenario generator: more threat centers, more no-fly patches, tighter energy budget
def generate_hard_scenario(sid):
    rng = np.random.default_rng(5000 + sid)
    grid = GRID
    threat = rng.beta(2.2, 5.0, size=(grid, grid)) * 0.35
    comm_loss = rng.beta(2.0, 5.5, size=(grid, grid)) * 0.30
    weather = rng.beta(2.0, 7.0, size=(grid, grid)) * 0.20
    no_fly = np.zeros((grid, grid), dtype=int)

    for _ in range(11):  # was 6
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(2.0, 5.2)
        amp = rng.uniform(0.45, 0.95)
        threat += amp * m.gaussian_blob(grid, cx, cy, sigma)

    for _ in range(5):
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(2.5, 6.0)
        amp = rng.uniform(0.35, 0.85)
        comm_loss += amp * m.gaussian_blob(grid, cx, cy, sigma)

    for _ in range(3):
        cx, cy = rng.integers(4, grid - 4, size=2)
        sigma = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.20, 0.50)
        weather += amp * m.gaussian_blob(grid, cx, cy, sigma)

    threat = np.clip(threat, 0, 1)
    comm_loss = np.clip(comm_loss, 0, 1)
    weather = np.clip(weather, 0, 1)

    for _ in range(14):  # was 5 -- dense no-fly patches to fragment corridors
        cx, cy = rng.integers(4, grid - 4, size=2)
        radius = rng.integers(2, 5)  # was 1-3
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

    energy_budget = int(rng.integers(45, 65))  # was 65-95, tightened
    risk_threshold = float(rng.uniform(0.25, 0.42))
    comm_threshold = float(rng.uniform(0.25, 0.45))

    return {
        "scenario_id": sid, "threat": threat, "comm_loss": comm_loss, "weather": weather,
        "no_fly": no_fly, "start": start, "goal": goal, "energy_budget": energy_budget,
        "risk_threshold": risk_threshold, "comm_threshold": comm_threshold,
    }

results = []
for sid in range(60):
    scenario = generate_hard_scenario(sid)
    path, _ = qkg_qi_search(scenario)
    metr = path_metrics(path, scenario)
    results.append(metr)

successes = [r["success"] for r in results]
fails = [r for r in results if r["success"] == 0]
print(f"Stress test: {sum(successes)}/{len(successes)} succeeded, {len(fails)} failed")
for i, r in enumerate(results):
    if r["success"] == 0:
        print(f"  fail #{i}: path_len={r['path_length']}, energy={r['energy_cost']}, "
              f"risk={r['avg_path_risk']}, viol={r['constraint_violations']}")
