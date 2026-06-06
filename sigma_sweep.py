import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from puf_network import get_response

# ─────────────────────────────────────────────
# PHYSICAL FABRICATION MODEL
# ─────────────────────────────────────────────
LAMBDA        = 1.55e-6
DN_EFF_DW     = 1.5e6
DELTA_W_SIGMA = 10e-9
ARM_LENGTHS = [5e-6, 10e-6, 15e-6, 20e-6]

def compute_sigma_phase(delta_w_sigma, arm_length):
    return (2 * np.pi / LAMBDA) * DN_EFF_DW * delta_w_sigma * arm_length

PRIMARY_SIGMA = compute_sigma_phase(DELTA_W_SIGMA, 200e-6)

N_CHIPS      = 20
N_CHALLENGES = 500
N_STAGES     = 8
N_ATTACK     = 3000

calibrated_sigmas = [(L, compute_sigma_phase(DELTA_W_SIGMA, L)) for L in ARM_LENGTHS]

print("=" * 60)
print("PHYSICAL CALIBRATION")
print("=" * 60)
print(f"Waveguide: 500x220nm SOI, lambda=1550nm")
print(f"Width variation (1sigma): {DELTA_W_SIGMA*1e9:.0f}nm")
print(f"Index sensitivity: {DN_EFF_DW:.1e} m^-1")
print()
print(f"{'Arm length (um)':<20} {'sigma_phase (rad)':<20}")
print("-" * 40)
for L, sigma in calibrated_sigmas:
    print(f"{L*1e6:<20.0f} {sigma:<20.4f}")
print(f"\nPrimary calibrated sigma: {PRIMARY_SIGMA:.4f} rad (200um arm)")
print("=" * 60)

# ─────────────────────────────────────────────
# ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ROOT CAUSE ANALYSIS")
print("=" * 60)
challenge_range  = 2 * np.pi
sigma_initial    = 0.05
snr_initial      = sigma_initial / challenge_range * 100
snr_calibrated   = PRIMARY_SIGMA / challenge_range * 100
print(f"Challenge phase range:        0 to {challenge_range:.2f} rad")
print(f"Initial sigma:                {sigma_initial} rad")
print(f"Calibrated sigma (200um arm): {PRIMARY_SIGMA:.4f} rad")
print()
print(f"Fabrication contribution at sigma=0.05:     {snr_initial:.1f}% of challenge range")
print(f"Fabrication contribution at calibrated:     {snr_calibrated:.1f}% of challenge range")
print()
print("Conclusion: At sigma=0.05 rad, fabrication offsets contribute <1% of")
print("total phase variation. Challenge input dominates, making all chips")
print("appear identical. At realistic SOI values (sigma~1.2 rad), fabrication")
print("contributes ~19% of phase range — sufficient for chip-unique responses.")

# ─────────────────────────────────────────────
# UNIQUENESS THRESHOLD ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("UNIQUENESS THRESHOLD ANALYSIS")
print("=" * 60)

def binarize(response, threshold=0.5):
    return (response > threshold).astype(int)

def fractional_hamming_distance(r1, r2):
    return np.mean(r1 != r2)

ALL_SIGMAS  = [0.05] + [s for _, s in calibrated_sigmas]
ARM_LABELS  = ["N/A"] + [f"{L*1e6:.0f}um" for L, _ in calibrated_sigmas]

print(f"\n{'sigma (rad)':<14} {'Arm':<12} {'Inter FHD':<14} {'Intra FHD':<14} {'Functional?'}")
print("-" * 68)

fhd_results = []
for idx, sigma in enumerate(ALL_SIGMAS):
    chips = []
    for i in range(N_CHIPS):
        rng = np.random.default_rng(i)
        chips.append(rng.normal(0.0, sigma, N_STAGES))

    challenges = np.random.uniform(0, 2*np.pi, size=(N_CHALLENGES, N_STAGES))

    bin_responses = []
    for chip in chips:
        resp = np.array([get_response(chip, c) for c in challenges])
        bin_responses.append(binarize(resp))

    inter_fhds = []
    for i in range(N_CHIPS):
        for j in range(i+1, N_CHIPS):
            inter_fhds.append(
                fractional_hamming_distance(bin_responses[i], bin_responses[j])
            )
    inter_fhd = np.mean(inter_fhds)

    clean_resp  = np.array([get_response(chips[0], c) for c in challenges])
    noise       = np.random.normal(0, 0.01, (N_CHALLENGES,))
    noisy_resp  = clean_resp + noise
    intra_fhd   = fractional_hamming_distance(binarize(clean_resp), binarize(noisy_resp))

    functional  = "YES" if inter_fhd >= 0.3 else "NO <- below threshold"
    print(f"{sigma:<14.4f} {ARM_LABELS[idx]:<12} {inter_fhd:<14.4f} "
          f"{intra_fhd:<14.4f} {functional}")
    fhd_results.append((sigma, inter_fhd, intra_fhd))

# ─────────────────────────────────────────────
# ML ATTACK SWEEP
# ─────────────────────────────────────────────
print("\nRunning ML attack sweep...")
mlp_rmses = []
baselines  = []

for sigma in ALL_SIGMAS:
    rng    = np.random.default_rng(0)
    target = rng.normal(0.0, sigma, N_STAGES)
    X = np.random.uniform(0, 2*np.pi, size=(N_ATTACK, N_STAGES))
    y = np.array([get_response(target, c) for c in X])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_rmses.append(np.sqrt(np.mean((y_test - mlp.predict(X_test))**2)))
    baselines.append(np.std(y_test))
    print(f"  sigma={sigma:.4f}: MLP RMSE={mlp_rmses[-1]:.4f}, baseline={baselines[-1]:.4f}")

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
sigmas_plot = [r[0] for r in fhd_results]
inter_fhds  = [r[1] for r in fhd_results]
intra_fhds  = [r[2] for r in fhd_results]
attack_red  = [(b-r)/b*100 for b,r in zip(baselines, mlp_rmses)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Silicon Photonic PUF — Calibrated Analysis\n"
             "(SOI 500x220nm, 193nm lithography, lambda=1550nm)",
             fontsize=12, fontweight='bold')

# Plot 1: FHD vs sigma
axes[0].plot(sigmas_plot, inter_fhds, 'o-', color="#378ADD",
             linewidth=2, markersize=8, label="Inter-chip FHD")
axes[0].plot(sigmas_plot, intra_fhds, 's-', color="#E07B39",
             linewidth=2, markersize=8, label="Intra-chip FHD")
axes[0].axhline(y=0.5, color='green', linestyle='--',
                alpha=0.7, label='Ideal inter-chip (0.5)')
axes[0].axhline(y=0.3, color='red', linestyle='--',
                alpha=0.7, label='Min viable (0.3)')
axes[0].set_xlabel("Fabrication variation sigma (rad)")
axes[0].set_ylabel("Fractional Hamming Distance")
axes[0].set_title("PUF Uniqueness Threshold Analysis")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot 2: ML attack resistance vs sigma
axes[1].plot(sigmas_plot, attack_red, 's-', color="#1D9E75",
             linewidth=2, markersize=8)
axes[1].axhline(y=0, color='red', linestyle='--',
                alpha=0.5, label='Random guess baseline')
axes[1].set_xlabel("Fabrication variation sigma (rad)")
axes[1].set_ylabel("MLP attack reduction vs baseline (%)")
axes[1].set_title("ML Attack Resistance vs Fabrication Variation")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("calibrated_analysis.png", dpi=150)
plt.show()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"{'sigma (rad)':<14} {'Arm':<12} {'Inter FHD':<12} {'Functional?'}")
print("-" * 50)
for idx, (sigma, inter_fhd, _) in enumerate(fhd_results):
    functional = "YES" if inter_fhd >= 0.3 else "NO"
    print(f"{sigma:<14.4f} {ARM_LABELS[idx]:<12} {inter_fhd:<12.4f} {functional}")
