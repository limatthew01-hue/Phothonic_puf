import numpy as np
import matplotlib.pyplot as plt
from puf_network import simulate_chip, get_response

N_CHIPS      = 20
N_CHALLENGES = 500
N_STAGES     = 8

# Simulate 20 chips with different fabrication variations
chips = [simulate_chip(n_stages=N_STAGES, seed=i) for i in range(N_CHIPS)]

# Same challenges applied to every chip
shared_challenges = np.random.uniform(0, 2*np.pi, size=(N_CHALLENGES, N_STAGES))

# Collect all responses: shape (20 chips, 500 challenges)
print("Simulating 20 chips...")
all_responses = np.array([
    [get_response(chip, c) for c in shared_challenges]
    for chip in chips
])
print("Done.")

# --- Inter-chip variation ---
# How differently do chips respond to the same challenge?
# High = chips are unique = good PUF
inter_var = np.std(all_responses, axis=0).mean()

# --- Intra-chip reliability ---
# How consistent is one chip across repeated measurements?
# Low = chip is stable = good PUF
noise     = np.random.normal(0, 0.01, all_responses.shape)
intra_var = np.std(noise, axis=1).mean()

print(f"\nInter-chip variation (uniqueness):  {inter_var:.4f}  ← want HIGH")
print(f"Intra-chip variation (reliability): {intra_var:.4f}  ← want LOW")

# --- Plot 1: Inter-chip variation distribution ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(np.std(all_responses, axis=0), bins=30, color="#378ADD", edgecolor="white")
plt.xlabel("Std dev across chips")
plt.ylabel("Count")
plt.title("Inter-chip variation")

# --- Plot 2: Response heatmap ---
plt.subplot(1, 2, 2)
plt.imshow(all_responses, aspect="auto", cmap="viridis")
plt.colorbar(label="Response power")
plt.xlabel("Challenge index")
plt.ylabel("Chip index")
plt.title("Response heatmap (chips × challenges)")

plt.tight_layout()
plt.savefig("puf_metrics.png", dpi=150)
plt.show()