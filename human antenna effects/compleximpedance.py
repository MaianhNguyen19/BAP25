import numpy as np
import matplotlib.pyplot as plt

# Cole–Cole parameters for muscle (4 dispersions) and dry skin (2 dispersions)
tissues = {
    "Muscle": {
        "eps_inf": 4.0,
        "terms": [
            (50.0,    7.23e-12, 0.10),
            (7000.0,  353.68e-9, 0.10),
            (1.2e6,   318.31e-6, 0.00),
            (2.5e7,   2.274e-3,  0.00),
        ],
        "sigma": 0.20  # S/m
    },
    "Dry Skin": {
        "eps_inf": 4.0,
        "terms": [
            (32.0,    7.23e-12, 0.00),
            (1100.0,  32.48e-9, 0.20),
        ],
        "sigma": 0.0002  # S/m
    }
}

# Frequency range: 1 Hz to 1 GHz
freq = np.logspace(0, 9, 1000)  # decades from 10^0 to 10^9 Hz
omega = 2 * np.pi * freq

# Physical constants
eps0 = 8.854e-12  # Vacuum permittivity (F/m)

def cole_cole(eps_inf, terms, omega, sigma):
    eps = eps_inf * np.ones_like(omega, dtype=complex)
    for delta_eps, tau, alpha in terms:
        eps += delta_eps / (1 + (1j * omega * tau)**(1 - alpha))
    eps -= 1j * sigma / (omega * eps0)
    return eps

# Cylinder model parameters
L = 0.30  # length of segment (m)
r = 0.03  # radius of cylinder (m)
A = np.pi * r**2  # cross-sectional area (m^2)

# Compute impedance for each tissue
impedances = {}
phases = {}
for name, params in tissues.items():
    eps_c = cole_cole(params["eps_inf"], params["terms"], omega, params["sigma"])
    sigma_comp = 1j * omega * eps0 * eps_c
    Z = L / (A * sigma_comp)
    impedances[name] = np.abs(Z)
    phases[name] = np.angle(Z, deg=True)

# Plot magnitude on its own page
plt.figure(figsize=(8, 6))
for name in tissues:
    plt.loglog(freq, impedances[name], label=f"{name} |Z| (Ω)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Impedance magnitude |Z| (Ω)")
plt.title("Body Impedance Magnitude vs Frequency")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Plot phase on a separate page
plt.figure(figsize=(8, 6))
for name in tissues:
    plt.semilogx(freq, phases[name], label=f"{name} ∠Z (°)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Impedance phase ∠Z (°)")
plt.title("Body Impedance Phase vs Frequency")
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()