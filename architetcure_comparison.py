import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from puf_network import (get_response, mzi_transfer,
                          clements_mesh_response,
                          butterfly_response,
                          ring_resonator_response)

SIGMA        = 0.30
N_CHIPS      = 20
N_CHALLENGES = 500
N_ATTACK     = 3000

def binarize(r):
    return (r > 0.5).astype(int)

def fhd(a, b):
    return float(np.mean(a != b))

# ─────────────────────────────────────────────
# PYTORCH ATTACK MODELS
# ─────────────────────────────────────────────
class LSTMAttack(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class TransformerAttack(nn.Module):
    def __init__(self, n_stages=8, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                          dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        x = self.transformer(self.embed(x))
        return self.fc(x.mean(dim=1)).squeeze(1)

def train_torch(model, X, y, epochs=150, lr=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X).unsqueeze(-1),
                      torch.FloatTensor(y)),
        batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model

def eval_torch(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X).unsqueeze(-1)).numpy()
    return float(np.sqrt(np.mean((y - preds)**2)))

# ─────────────────────────────────────────────
# HYBRID ARCHITECTURE: Butterfly + random couplers
# Second source of randomness beyond phase offsets
# ─────────────────────────────────────────────
def butterfly_hybrid_response(offsets, challenge, coupler_vars):
    """
    Butterfly network with randomized coupler ratios.
    coupler_vars: per-MZI coupling deviation from 0.5
    Adds a second independent source of physical randomness.
    """
    N_stages = len(offsets)
    N_modes  = 2 ** (N_stages // 4 + 1)
    field    = np.zeros(N_modes, dtype=complex)
    field[0] = 1.0
    for stage in range(N_stages):
        new_field = field.copy()
        stride    = max(1, (2**(stage % (N_modes//2).bit_length())) % N_modes)
        for i in range(0, N_modes - stride, stride * 2):
            phase   = challenge[stage] + offsets[stage]
            # Randomized coupler: κ = 0.5 + coupler variation
            kappa   = np.clip(0.5 + coupler_vars[stage], 0.1, 0.9)
            tau     = np.sqrt(1 - kappa)
            kap     = np.sqrt(kappa)
            dc      = (1/np.sqrt(2)) * np.array([[tau,  1j*kap],
                                                  [1j*kap, tau]])
            ps      = np.array([[np.exp(1j*phase), 0],[0, 1]])
            mzi     = dc @ ps @ dc
            pair    = np.array([field[i], field[i+stride]])
            result  = mzi @ pair
            new_field[i]          = result[0]
            new_field[i + stride] = result[1]
        field = new_field
    return float(abs(field[0])**2)

# ─────────────────────────────────────────────
# BENCHMARK FUNCTION — tests all 3 attackers
# ─────────────────────────────────────────────
def full_benchmark(label, get_resp, make_offsets, make_challenge, n_dim):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    # FHD
    chips      = [make_offsets(i) for i in range(N_CHIPS)]
    challenges = [make_challenge() for _ in range(N_CHALLENGES)]
    bin_resps  = []
    for chip in chips:
        resp = np.array([get_resp(chip, c) for c in challenges])
        bin_resps.append(binarize(resp))

    inter_fhds = [fhd(bin_resps[i], bin_resps[j])
                  for i in range(N_CHIPS)
                  for j in range(i+1, N_CHIPS)]
    inter_fhd  = float(np.mean(inter_fhds))
    clean  = np.array([get_resp(chips[0], c) for c in challenges])
    noisy  = clean + np.random.normal(0, 0.01, clean.shape)
    intra_fhd = fhd(binarize(clean), binarize(noisy))
    functional = "YES" if inter_fhd >= 0.3 else "NO"
    print(f"  Inter FHD: {inter_fhd:.4f}  "
          f"Intra FHD: {intra_fhd:.4f}  Functional: {functional}")

    # Attack dataset
    X_flat = np.array([make_challenge().flatten()
                        for _ in range(N_ATTACK)])
    y      = np.array([get_resp(chips[0],
                                 x.reshape(make_challenge().shape))
                        for x in X_flat])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat, y, test_size=0.2, random_state=42)
    baseline = float(np.std(y_te))

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64,64),
                        max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    mlp_rmse = float(np.sqrt(np.mean(
        (y_te - mlp.predict(X_te))**2)))
    mlp_red  = (baseline - mlp_rmse) / baseline * 100

    # LSTM
    lstm = LSTMAttack()
    train_torch(lstm, X_tr, y_tr)
    lstm_rmse = eval_torch(lstm, X_te, y_te)
    lstm_red  = (baseline - lstm_rmse) / baseline * 100

    # Transformer
    tfm = TransformerAttack(n_stages=n_dim)
    train_torch(tfm, X_tr, y_tr)
    tfm_rmse = eval_torch(tfm, X_te, y_te)
    tfm_red  = (baseline - tfm_rmse) / baseline * 100

    print(f"  Baseline RMSE: {baseline:.4f}")
    print(f"  MLP:         RMSE={mlp_rmse:.4f}  "
          f"attack reduction={mlp_red:.1f}%")
    print(f"  LSTM:        RMSE={lstm_rmse:.4f}  "
          f"attack reduction={lstm_red:.1f}%")
    print(f"  Transformer: RMSE={tfm_rmse:.4f}  "
          f"attack reduction={tfm_red:.1f}%")

    return (inter_fhd, intra_fhd,
            mlp_red, lstm_red, tfm_red, functional)

# ─────────────────────────────────────────────
# RUN ALL ARCHITECTURES
# ─────────────────────────────────────────────
print("FULL ARCHITECTURE × ATTACK BENCHMARK")
print(f"σ={SIGMA} rad | {N_CHIPS} chips | {N_CHALLENGES} challenges")

results = {}

# 1. Linear N=8
results["Linear N=8"] = full_benchmark(
    "Linear chain N=8",
    get_response,
    lambda s: np.random.default_rng(s).normal(0, SIGMA, 8),
    lambda: np.random.uniform(0, 2*np.pi, 8),
    8)

# 2. Clements 4×4
N_M, N_L = 4, 4
N_P = N_M // 2
results["Clements 4×4"] = full_benchmark(
    "Clements mesh 4×4",
    lambda o, c: clements_mesh_response(o, c, N=N_M),
    lambda s: np.random.default_rng(s).normal(0, SIGMA, (N_L, N_P)),
    lambda: np.random.uniform(0, 2*np.pi, (N_L, N_P)),
    N_L * N_P)

# 3. Butterfly N=8
results["Butterfly N=8"] = full_benchmark(
    "Butterfly N=8",
    butterfly_response,
    lambda s: np.random.default_rng(s).normal(0, SIGMA, 8),
    lambda: np.random.uniform(0, 2*np.pi, 8),
    8)

# 4. Butterfly hybrid (new architecture)
results["Butterfly hybrid"] = full_benchmark(
    "Butterfly hybrid (phase + coupler variation)",
    lambda o, c: butterfly_hybrid_response(
        o[:8], c[:8],
        o[8:]),   # second half of offsets = coupler variations
    lambda s: np.concatenate([
        np.random.default_rng(s).normal(0, SIGMA, 8),
        np.random.default_rng(s+100).normal(0, 0.02, 8)]),
    lambda: np.random.uniform(0, 2*np.pi, 8),
    8)

# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY — Attack reduction % (negative = attack FAILS = MORE secure)")
print("="*80)
print(f"{'Architecture':<22} {'Inter FHD':<11} {'Functional':<11} "
      f"{'MLP %':<10} {'LSTM %':<10} {'Transformer %'}")
print("-"*80)
for label, (inter, intra, mlp, lstm, tfm, func) in results.items():
    print(f"{label:<22} {inter:<11.4f} {func:<11} "
          f"{mlp:<10.1f} {lstm:<10.1f} {tfm:.1f}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
# Remove hybrid from plot — implementation needs parameter fixes
plot_results = {k: v for k, v in results.items()
                if k != "Butterfly hybrid"}

labels     = list(plot_results.keys())
inter_fhds = [plot_results[l][0] for l in labels]
mlp_reds   = [plot_results[l][2] for l in labels]
lstm_reds  = [plot_results[l][3] for l in labels]
tfm_reds   = [plot_results[l][4] for l in labels]

x   = np.arange(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"Architecture × Attack Benchmark (σ={SIGMA} rad)\n"
             "Negative attack reduction = attack fails = more secure",
             fontsize=12, fontweight='bold')

ax1.bar(x, inter_fhds, 0.5, color="#378ADD", alpha=0.85)
ax1.axhline(0.5, color='green', linestyle='--',
            alpha=0.7, label='Ideal (0.5)')
ax1.axhline(0.3, color='red', linestyle='--',
            alpha=0.7, label='Min viable (0.3)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha='right')
ax1.set_ylabel("Inter-chip FHD")
ax1.set_title("PUF Uniqueness")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

w = 0.25
ax2.bar(x - w, mlp_reds,  w, color="#378ADD",
        alpha=0.85, label="MLP")
ax2.bar(x,     lstm_reds, w, color="#1D9E75",
        alpha=0.85, label="LSTM")
ax2.bar(x + w, tfm_reds,  w, color="#E07B39",
        alpha=0.85, label="Transformer")
ax2.axhline(0, color='red', linestyle='--', alpha=0.6)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=15, ha='right')
ax2.set_ylabel("Attack reduction vs baseline (%)")
ax2.set_title("ML Attack Resistance\n(all three attackers)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("full_benchmark.png", dpi=150)
plt.show()
