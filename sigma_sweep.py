import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from puf_network import simulate_chip, get_response

# Published SOI fabrication variation range
# Based on ±10-20nm width variation on 500nm waveguide, 200μm arm length
# Δφ = (2π/λ) × Δn_eff × L → σ range of 0.05 to 2.0 rad
SIGMA_VALUES = [0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
N_CHIPS      = 20
N_CHALLENGES = 500
N_STAGES     = 8
N_ATTACK     = 3000

inter_vars = []
mlp_rmses  = []
baselines  = []

for sigma in SIGMA_VALUES:
    print(f"\nRunning σ = {sigma:.2f} rad...")

    # --- Inter-chip variation ---
    chips = []
    for i in range(N_CHIPS):
        rng = np.random.default_rng(i)
        chips.append(rng.normal(0.0, sigma, N_STAGES))

    challenges = np.random.uniform(0, 2*np.pi, size=(N_CHALLENGES, N_STAGES))
    responses  = np.array([
        [get_response(chip, c) for c in challenges]
        for chip in chips
    ])
    inter_var = np.std(responses, axis=0).mean()
    inter_vars.append(inter_var)

    # --- ML attack at this σ ---
    target_chip = chips[0]
    X = np.random.uniform(0, 2*np.pi, size=(N_ATTACK, N_STAGES))
    y = np.array([get_response(target_chip, c) for c in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_rmse = np.sqrt(np.mean((y_test - mlp_pred)**2))
    baseline = np.std(y_test)

    mlp_rmses.append(mlp_rmse)
    baselines.append(baseline)
    print(f"  Inter-chip variation: {inter_var:.4f}")
    print(f"  MLP RMSE: {mlp_rmse:.4f}  |  Baseline: {baseline:.4f}")

# --- Plot results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Inter-chip variation vs sigma
ax1.plot(SIGMA_VALUES, inter_vars, 'o-', color="#378ADD", linewidth=2, markersize=8)
ax1.set_xlabel("Fabrication variation σ (rad)")
ax1.set_ylabel("Inter-chip variation (std dev)")
ax1.set_title("PUF Uniqueness vs Fabrication Variation")
ax1.axvspan(0.1, 1.0, alpha=0.1, color='green', label='Realistic SOI range')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: ML attack resistance vs sigma
attack_reduction = [(b - r)/b * 100 for b, r in zip(baselines, mlp_rmses)]
ax2.plot(SIGMA_VALUES, attack_reduction, 's-', color="#1D9E75", linewidth=2, markersize=8)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random guess baseline')
ax2.set_xlabel("Fabrication variation σ (rad)")
ax2.set_ylabel("MLP attack reduction vs baseline (%)")
ax2.set_title("ML Attack Success vs Fabrication Variation")
ax2.axvspan(0.1, 1.0, alpha=0.1, color='green', label='Realistic SOI range')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sigma_sweep.png", dpi=150)
plt.show()

# Print summary table
print("\n--- Summary ---")
print(f"{'σ (rad)':<12} {'Inter-chip var':<18} {'MLP RMSE':<12} {'Attack reduction'}")
print("-" * 58)
for i, sigma in enumerate(SIGMA_VALUES):
    print(f"{sigma:<12.2f} {inter_vars[i]:<18.4f} {mlp_rmses[i]:<12.4f} {attack_reduction[i]:.1f}%")