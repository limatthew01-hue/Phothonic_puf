import numpy as np
import matplotlib.pyplot as plt

def mzi_transfer(phase):
    # Correct 50/50 directional coupler — prefactor is 1/sqrt(2)
    dc = (1/np.sqrt(2)) * np.array([[1,  1j],
                                     [1j,  1]])
    # Phase shifter on arm 0
    ps = np.array([[np.exp(1j * phase), 0],
                   [0,                  1]])
    # Full MZI: DC2 @ PS @ DC1
    mzi = dc @ ps @ dc
    # Input at port 0, read output at port 1
    input_field = np.array([1, 0])
    output = mzi @ input_field
    return abs(output[1])**2   # ← port 1, not port 0

phases       = np.linspace(0, 2 * np.pi, 200)
power_sim    = [mzi_transfer(p) for p in phases]
power_theory = np.cos(phases / 2)**2

plt.figure(figsize=(8, 4))
plt.plot(phases, power_sim,    label="Matrix simulation", linewidth=2)
plt.plot(phases, power_theory, label="Theory cos²(Δφ/2)", linestyle="--", linewidth=2)
plt.xlabel("Phase (rad)")
plt.ylabel("Output power")
plt.legend()
plt.title("MZI sanity check")
plt.tight_layout()
plt.savefig("mzi_sanity_check.png", dpi=150)
plt.show()