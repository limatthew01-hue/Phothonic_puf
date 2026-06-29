import numpy as np
import matplotlib.pyplot as plt

def mzi_transfer(phase):
    dc = (1/np.sqrt(2)) * np.array([[1,  1j],
                                     [1j,  1]])
    ps = np.array([[np.exp(1j * phase), 0],
                   [0,                  1]])
    mzi = dc @ ps @ dc
    return mzi

def simulate_chip(n_stages=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=0.05, size=n_stages)

def get_response(offsets, challenge):
    field = np.array([1, 0])
    for offset, c in zip(offsets, challenge):
        total_phase = c + offset
        mzi = mzi_transfer(total_phase)
        field = mzi @ field
    return abs(field[1])**2

if __name__ == "__main__":
    n_stages = 8
    chip = simulate_chip(n_stages=n_stages, seed=0)
    challenges = np.random.uniform(0, 2*np.pi, size=(1000, n_stages))
    responses  = np.array([get_response(chip, c) for c in challenges])

    print(f"Generated {len(responses)} CRPs")
    print(f"Response range: {responses.min():.3f} to {responses.max():.3f}")

    plt.figure(figsize=(7, 4))
    plt.hist(responses, bins=40, color="#378ADD", edgecolor="white")
    plt.xlabel("Response power")
    plt.ylabel("Count")
    plt.title("CRP response distribution — single chip")
    plt.tight_layout()
    plt.savefig("crp_distribution.png", dpi=150)
    plt.show()
# ─────────────────────────────────────────────
# NEW ARCHITECTURE DEFINITIONS
# ─────────────────────────────────────────────

def clements_mesh_response(offsets, challenge, N=4):
    N_layers = N
    field = np.zeros(N, dtype=complex)
    field[0] = 1.0
    for layer in range(N_layers):
        new_field = field.copy()
        start = 0 if layer % 2 == 0 else 1
        for col in range(start, N - 1, 2):
            idx   = col // 2
            phase = challenge[layer, idx] + offsets[layer, idx]
            mzi   = mzi_transfer(phase)
            pair  = np.array([field[col], field[col + 1]])
            result = mzi @ pair
            new_field[col]     = result[0]
            new_field[col + 1] = result[1]
        field = new_field
    return float(abs(field[0]) ** 2)


def butterfly_response(offsets, challenge, N_stages=8):
    N_modes = 2 ** (N_stages // 4 + 1)
    field   = np.zeros(N_modes, dtype=complex)
    field[0] = 1.0
    for stage in range(N_stages):
        new_field = field.copy()
        stride    = max(1, (2 ** (stage % (N_modes // 2).bit_length())) % N_modes)
        for i in range(0, N_modes - stride, stride * 2):
            phase  = challenge[stage] + offsets[stage]
            mzi    = mzi_transfer(phase)
            pair   = np.array([field[i], field[i + stride]])
            result = mzi @ pair
            new_field[i]          = result[0]
            new_field[i + stride] = result[1]
        field = new_field
    return float(abs(field[0]) ** 2)


def ring_resonator_response(offsets, challenge):
    transmission = 1.0
    for offset, c in zip(offsets, challenge):
        delta      = c + offset
        Q          = 10000
        kappa      = 1.0 / Q
        T_through  = (delta ** 2) / (delta ** 2 + kappa ** 2)
        transmission *= T_through
    return float(np.clip(transmission, 0, 1))
