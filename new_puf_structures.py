# Arms Race Step 2: New PUF Structures
# Tests linear chain (N=8,16,32) and 2D mesh topology
# Measures FHD and ML attack resistance for each

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from puf_network import mzi_transfer

SIGMA        = 0.30   # fixed from literature
N_CHIPS      = 20
N_CHALLENGES = 500
N_ATTACK     = 3000

def binarize(resp):
    return (resp > 0.5).astype(int)

def fhd(r1, r2):
    return np.mean(r1 != r2)

# ─────────────────────────────────────────────
# STRUCTURE 1: Linear chain (variable N stages)
# ─────────────────────────────────────────────
def get_response_linear(offsets, challenge):
    field = np.array([1, 0], dtype=complex)
    for offset, c in zip(offsets, challenge):
        field = mzi_transfer(c + offset) @ field
    return float(abs(field[1]) ** 2)


# ─────────────────────────────────────────────
# STRUCTURE 2: 2D mesh topology
# Alternating layers: even rows couple (0,1),(2,3)...
# Odd rows couple (1,2),(3,4)...
# Light takes multiple paths → harder to model
# ─────────────────────────────────────────────
def get_response_2d(offsets_2d, challenge_2d):
    """
    offsets_2d:   (N_rows, N_cols) fabrication offsets
    challenge_2d: (N_rows, N_cols) applied phases
    """
    N_rows, N_cols = offsets_2d.shape
    fields = np.zeros(N_cols, dtype=complex)
    fields[0] = 1.0

    for row in range(N_rows):
        new_fields = fields.copy()
        # Alternate which pairs couple each row
        start = 0 if row % 2 == 0 else 1
        for col in range(start, N_cols - 1, 2):
            phase  = challenge_2d[row, col] + offsets_2d[row, col]
            result = mzi_transfer(phase) @ np.array([fields[col],
                                                      fields[col + 1]])
            new_fields[col]     = result[0]
            new_fields[col + 1] = result[1]
        fields = new_fields

    return float(abs(fields[0]) ** 2)


# ─────────────────────────────────────────────
# GENERIC BENCHMARK FUNCTION
# ─────────────────────────────────────────────
def benchmark_structure(label, get_resp_fn, n_offsets,
                         challenge_dim, sigma=SIGMA):
    print(f"\n  Benchmarking: {label}")

    # Generate chips
    chips = []
    for i in range(N_CHIPS):
        rng = np.random.default_rng(i)
        chips.append(rng.normal(0.0, sigma, n_offsets))

    # FHD
    challenges = np.random.uniform(0, 2*np.pi,
                                   size=(N_CHALLENGES, challenge_dim))
    bin_resps = []
    for chip in chips:
        resp = np.array([get_resp_fn(chip, c) for c in challenges])
        bin_resps.append(binarize(resp))

    inter_fhds = [
        fhd(bin_resps[i], bin_resps[j])
        for i in range(N_CHIPS)
        for j in range(i + 1, N_CHIPS)
    ]
    inter_fhd = float(np.mean(inter_fhds))

    # Intra-chip reliability
    clean = np.array([get_resp_fn(chips[0], c) for c in challenges])
    noisy = clean + np.random.normal(0, 0.01, clean.shape)
    intra_fhd = float(fhd(binarize(clean), binarize(noisy)))

    # ML attack
    X = np.random.uniform(0, 2*np.pi, size=(N_ATTACK, challenge_dim))
    y = np.array([get_resp_fn(chips[0], c) for c in X])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64),
                       max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    mlp_rmse = float(np.sqrt(np.mean((y_te - mlp.predict(X_te)) ** 2)))
    baseline  = float(np.std(y_te))

    functional = "YES" if inter_fhd >= 0.3 else "NO"
    print(f"    Inter-chip FHD: {inter_fhd:.4f}  "
          f"Intra-chip FHD: {intra_fhd:.4f}  "
          f"Functional: {functional}")
    print(f"    MLP RMSE: {mlp_rmse:.4f}  Baseline: {baseline:.4f}  "
          f"Attack reduction: {(baseline-mlp_rmse)/baseline*100:.1f}%")

    return inter_fhd, intra_fhd, mlp_rmse, baseline


# ─────────────────────────────────────────────
# RUN ALL STRUCTURES
# ─────────────────────────────────────────────
print("=" * 60)
print("STRUCTURE COMPARISON")
print(f"Fixed σ = {SIGMA} rad")
print("=" * 60)

results = {}

# Linear chain: N = 8, 16, 32
for n in [8, 16, 32]:
    label = f"Linear chain N={n}"
    inter, intra, rmse, base = benchmark_structure(
        label=label,
        get_resp_fn=get_response_linear,
        n_offsets=n,
        challenge_dim=n
    )
    results[label] = (inter, intra, rmse, base)

# 2D mesh: 4×4 (16 MZIs, 16-dim challenge)
N_ROWS, N_COLS = 4, 4

def get_response_2d_flat(offsets_flat, challenge_flat):
    return get_response_2d(
        offsets_flat.reshape(N_ROWS, N_COLS),
        challenge_flat.reshape(N_ROWS, N_COLS)
    )

label = f"2D mesh {N_ROWS}x{N_COLS}"
inter, intra, rmse, base = benchmark_structure(
    label=label,
    get_resp_fn=get_response_2d_flat,
    n_offsets=N_ROWS * N_COLS,
    challenge_dim=N_ROWS * N_COLS
)
results[label] = (inter, intra, rmse, base)

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)
print(f"{'Structure':<22} {'Inter FHD':<12} {'Intra FHD':<12} "
      f"{'MLP RMSE':<12} {'Functional?'}")
print("-" * 72)
for label, (inter, intra, rmse, base) in results.items():
    functional = "YES" if inter >= 0.3 else "NO"
    print(f"{label:<22} {inter:<12.4f} {intra:<12.4f} "
          f"{rmse:<12.4f} {functional}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
labels      = list(results.keys())
inter_fhds  = [results[l][0] for l in labels]
intra_fhds  = [results[l][1] for l in labels]
mlp_rmses   = [results[l][2] for l in labels]
baselines   = [results[l][3] for l in labels]
attack_reds = [(b - r) / b * 100
               for b, r in zip(baselines, mlp_rmses)]

x = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"PUF Structure Comparison (σ={SIGMA} rad)",
             fontsize=13, fontweight='bold')

# FHD comparison
ax1.bar(x - 0.2, inter_fhds, 0.35, color="#378ADD",
        label="Inter-chip FHD", alpha=0.85)
ax1.bar(x + 0.2, intra_fhds, 0.35, color="#E07B39",
        label="Intra-chip FHD", alpha=0.85)
ax1.axhline(y=0.5, color='green', linestyle='--',
            alpha=0.6, label='Ideal inter-chip (0.5)')
ax1.axhline(y=0.3, color='red', linestyle='--',
            alpha=0.6, label='Min viable (0.3)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha='right')
ax1.set_ylabel("Fractional Hamming Distance")
ax1.set_title("Uniqueness and Reliability")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# Attack resistance comparison
colors = ["#378ADD" if r > 0 else "#E07B39" for r in attack_reds]
ax2.bar(x, attack_reds, 0.5, color=colors, alpha=0.85)
ax2.axhline(y=0, color='red', linestyle='--',
            alpha=0.6, label='Random guess baseline')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_ylabel("MLP attack reduction vs baseline (%)")
ax2.set_title("ML Attack Resistance\n(negative = attack worse than random)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("puf_structures.png", dpi=150)
plt.show()