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