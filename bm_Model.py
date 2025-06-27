# Dimensionless Bistritzer-MacDonald Model for Twisted Bilayer Graphene
# Geometry, basis setup, dimensionless Hamiltonian, band plotting, validation

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- Physical constants and input parameters ---
a = 2.46e-10             # graphene lattice constant [m]
v_F = 1e6                # Fermi velocity [m/s]
hbar = 6.582119569e-16   # Planck's constant over 2π [eV·s]

# Twist angle
theta_deg = 1.1
theta = np.deg2rad(theta_deg)

# Tunneling amplitudes (eV), realistic ratio for relaxation
w0 = 0.0784  # AA coupling
w1 = 0.110   # AB/BA coupling

# --- Derived moiré scales ---
k_theta = (4*np.pi/(3*a)) * 2*np.sin(theta/2)  # moiré momentum scale [1/m]
E0 = hbar * v_F * k_theta                      # energy scale [eV]
print(f"Energy scale E0 = {E0:.3f} eV")

# Dimensionless tunneling
w0_dim = w0 / E0
w1_dim = w1 / E0

# Rotation matrix
def R(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

# Dirac K-point magnitude and dimensionless q vectors
K_mag = 4*np.pi / (3*a)
q1 = (R(-theta/2) @ np.array([0, K_mag]) - R(+theta/2) @ np.array([0, K_mag])) / k_theta
q2 = (R(-theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q3 = (R(-theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q_vecs = [q1, q2, q3]

# Pauli matrices
sigma0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])

# Tunneling matrices
phis = [np.arctan2(q[1], q[0]) for q in q_vecs]
T_mats = [w0_dim * sigma0 + w1_dim * (np.cos(phi) * sigma_x + np.sin(phi) * sigma_y) for phi in phis]

# Moiré reciprocal vectors
L_m = a / (2 * np.sin(theta / 2))                 # moiré period [m]
Gm_phys = 4 * np.pi / (np.sqrt(3) * L_m)          # physical moiré G magnitude [1/m]

def generate_G_vectors_physical(shells=3):
    b1 = Gm_phys * np.array([np.sqrt(3)/2, 1/2])
    b2 = Gm_phys * np.array([0, 1])
    Gs = []
    for i in range(-shells, shells+1):
        for j in range(-shells, shells+1):
            Gs.append(i*b1 + j*b2)
    return np.array(Gs)

G_vectors_phys = generate_G_vectors_physical(shells=3)
G_vectors = G_vectors_phys / k_theta
N = len(G_vectors)
print(f"Dimensionless basis size N = {N}")

# Hamiltonian builder
def build_Hk(k_vec, G_vectors):
    N = len(G_vectors)
    dim = 4 * N
    H = np.zeros((dim, dim), dtype=complex)
    for m, G in enumerate(G_vectors):
        idx = 4 * m
        k_top = R(+theta/2) @ k_vec - G
        k_bot = R(-theta/2) @ k_vec - G
        H0_top = k_top[0] * sigma_x + k_top[1] * sigma_y
        H0_bot = k_bot[0] * sigma_x + k_bot[1] * sigma_y
        H[idx:idx+2, idx:idx+2]     = H0_top
        H[idx+2:idx+4, idx+2:idx+4] = H0_bot
    for m, G in enumerate(G_vectors):
        for j, q in enumerate(q_vecs):
            target = G + q
            diffs = G_vectors - target[np.newaxis, :]
            dist = np.linalg.norm(diffs, axis=1)
            m2 = np.argmin(dist)
            if dist[m2] < 1e-6:
                i1, i2 = 4 * m, 4 * m2
                H[i1:i1+2,   i2+2:i2+4] = T_mats[j]
                H[i2+2:i2+4, i1:i1+2]   = T_mats[j].conj().T
    return H

def high_symmetry_path(num_k=60):
    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0, 1])
    Γ = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2 * b1 + b2) / 3
    path = []
    for start, end in [(Γ, K), (K, M), (M, Γ)]:
        for t in np.linspace(0, 1, num_k, endpoint=False):
            path.append((1 - t) * start + t * end)
    path.append(Γ)
    return np.array(path)

# Compute bands
k_path = high_symmetry_path()
bands_dim = np.zeros((len(k_path), 4*N))
for i, kpt in enumerate(k_path):
    ev, _ = eigh(build_Hk(kpt, G_vectors))
    bands_dim[i,:] = np.sort(np.real(ev))
bands = E0 * bands_dim  # convert to eV

# Plot
plt.figure(figsize=(8, 6))
for n in range(2*N-3, 2*N+1):
    plt.plot(bands[:, n], lw=1)
plt.xticks([0, 60, 120, 180], ["Γ", "K", "M", "Γ"])
plt.ylabel("Energy (eV)")
plt.title(f"TBG Bands (θ={theta_deg}°, shells=3)")
plt.grid(True)
plt.show()

# --- Validation ---
def flat_band_indices(N): return 2*N - 1, 2*N
def flat_band_bandwidth(bands):
    i1, i2 = flat_band_indices(N)
    return np.max(bands[:, i2] - bands[:, i1])
def gap_at_K(bands):
    i1, i2 = flat_band_indices(N)
    idx_K = len(k_path) // 3
    return bands[idx_K, i2+1] - bands[idx_K, i2]
def particle_hole_error(bands):
    errs = bands + bands[:, ::-1]
    return np.max(np.abs(errs))

# Print validation
bw   = flat_band_bandwidth(bands)
gapK = gap_at_K(bands)
symm = particle_hole_error(bands)
print(f"Flat-band bandwidth: {bw*1000:.1f} meV")
print(f"Gap above flat bands at K: {gapK*1000:.1f} meV")
print(f"Particle-hole symmetry error: {symm*1000:.1f} meV")
