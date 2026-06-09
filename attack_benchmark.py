# Arms Race Step 1: MLP vs LSTM vs Transformer
# Tests which model breaks the PUF with fewest CRPs
# Requires: pip3 install torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from puf_network import get_response

# ─────────────────────────────────────────────
# FIXED OPERATING POINT
# σ derived from SOI DUV lithography:
# σ_w = 5nm, L = 10μm, Δn_eff/Δw = 1.5μm⁻¹, λ = 1550nm
# σ = (2π/λ) × (Δn_eff/Δw) × σ_w × L = 0.30 rad
# Update this once Lao confirms foundry specs
# ─────────────────────────────────────────────
SIGMA    = 0.30
N_STAGES = 8
N_TOTAL  = 6000

print(f"Fixed operating point: σ = {SIGMA} rad, N = {N_STAGES} stages")

# Generate chip and full CRP dataset
rng   = np.random.default_rng(42)
chip  = rng.normal(0.0, SIGMA, N_STAGES)
X_all = np.random.uniform(0, 2*np.pi, size=(N_TOTAL, N_STAGES))
y_all = np.array([get_response(chip, c) for c in X_all])

# Fixed test set — never used for training
X_test = X_all[5000:]
y_test = y_all[5000:]
baseline = np.std(y_test)

CRP_SIZES = [100, 250, 500, 1000, 2500, 5000]

# ─────────────────────────────────────────────
# PYTORCH MODEL DEFINITIONS
# ─────────────────────────────────────────────

class LSTMAttack(nn.Module):
    """
    Treats challenge as ordered sequence of phase values.
    Each MZI stage is one time step.
    """
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, N_stages, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


class TransformerAttack(nn.Module):
    """
    Self-attention over challenge stages.
    Captures non-local dependencies between MZI phases.
    """
    def __init__(self, n_stages=8, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, N_stages, 1)
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)   # mean pool across stages
        return self.fc(x).squeeze(1)


def train_model(model, X_train, y_train, epochs=150, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_t = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def eval_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).unsqueeze(-1)).numpy()
    return float(np.sqrt(np.mean((y_test - preds) ** 2)))


# ─────────────────────────────────────────────
# BENCHMARK LOOP
# ─────────────────────────────────────────────
mlp_rmses  = []
lstm_rmses = []
tfm_rmses  = []

print(f"\nBaseline RMSE (random guess): {baseline:.4f}")
print(f"\n{'CRPs':<8} {'MLP':<12} {'LSTM':<12} {'Transformer':<12}")
print("-" * 48)

for n_crp in CRP_SIZES:
    X_train = X_all[:n_crp]
    y_train = y_all[:n_crp]

    # MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    mlp_rmse = float(np.sqrt(np.mean(
        (y_test - mlp.predict(X_test)) ** 2
    )))

    # LSTM
    lstm = LSTMAttack()
    train_model(lstm, X_train, y_train)
    lstm_rmse = eval_model(lstm, X_test, y_test)

    # Transformer
    tfm = TransformerAttack(n_stages=N_STAGES)
    train_model(tfm, X_train, y_train)
    tfm_rmse = eval_model(tfm, X_test, y_test)

    mlp_rmses.append(mlp_rmse)
    lstm_rmses.append(lstm_rmse)
    tfm_rmses.append(tfm_rmse)

    print(f"{n_crp:<8} {mlp_rmse:<12.4f} {lstm_rmse:<12.4f} {tfm_rmse:.4f}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(CRP_SIZES, mlp_rmses,  'o-', color="#378ADD",
         linewidth=2, markersize=8, label="MLP (baseline)")
plt.plot(CRP_SIZES, lstm_rmses, 's-', color="#1D9E75",
         linewidth=2, markersize=8, label="LSTM")
plt.plot(CRP_SIZES, tfm_rmses,  '^-', color="#E07B39",
         linewidth=2, markersize=8, label="Transformer")
plt.axhline(y=baseline, color='red', linestyle='--',
            alpha=0.7, label=f"Random baseline: {baseline:.4f}")
plt.xlabel("Number of CRPs (training set size)")
plt.ylabel("RMSE  (lower = stronger attack)")
plt.title(f"ML Attack Comparison: MLP vs LSTM vs Transformer\n"
          f"(σ={SIGMA} rad, N={N_STAGES} stages, fixed test set)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
plt.savefig("attack_benchmark.png", dpi=150)
plt.show()