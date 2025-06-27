# Complete Bistritzer-MacDonald Model with Validation Checks

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.linalg import eigh

# -- Physical Constants & Parameters --
hbar = 6.582119569e-16   # Planck's constant over 2π [eV·s]
v_F = 1e6                # Fermi velocity [m/s]
a = 2.46e-10             # Graphene lattice constant [m]

# Tunneling parameters (eV)
w0 = 0.08
w1 = 0.11

theta_deg = 1.1
theta = np.deg2rad(theta_deg)

# Moiré parameters
L_m = a / (2 * np.sin(theta/2))
Gm = 4 * pi / (np.sqrt(3) * L_m)

# Pauli matrices
sigma0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])

# Rotation
def R(angle): return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

# Dirac K-point magnitude & q vectors
K_mag = 4 * pi / (3 * a)
q1 = R(-theta/2) @ np.array([0, K_mag]) - R(+theta/2) @ np.array([0, K_mag])
q2 = R(-theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2])
q3 = R(-theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2])
q_vecs = [q1, q2, q3]

# Tunneling matrices T_j
phis = [np.arctan2(q[1], q[0]) for q in q_vecs]
T_mats = [w0 * sigma0 + w1 * (np.cos(phi)*sigma_x + np.sin(phi)*sigma_y) for phi in phis]

# Plane-wave basis

def generate_G_vectors(shells=3):
    b1 = Gm * np.array([np.sqrt(3)/2, 1/2])
    b2 = Gm * np.array([0, 1])
    Gs = []
    for i in range(-shells, shells+1):
        for j in range(-shells, shells+1):
            Gs.append(i*b1 + j*b2)
    return np.array(Gs)

G_vectors = generate_G_vectors(shells=3)
N = len(G_vectors)
print(f"Plane-wave basis size: N = {N}")

# Build Hamiltonian H(k)

def build_Hk(k_vec, G_vectors):
    N = len(G_vectors)
    dim = 4 * N
    H = np.zeros((dim, dim), dtype=complex)
    # intralayer
    for m, G in enumerate(G_vectors):
        idx = 4*m
        k_top = R(+theta/2) @ (k_vec - G)
        k_bot = R(-theta/2) @ (k_vec - G)
        H0_top = hbar*v_F*(k_top[0]*sigma_x + k_top[1]*sigma_y)
        H0_bot = hbar*v_F*(k_bot[0]*sigma_x + k_bot[1]*sigma_y)
        H[idx:idx+2, idx:idx+2] = H0_top
        H[idx+2:idx+4, idx+2:idx+4] = H0_bot
    # interlayer
    for m, G in enumerate(G_vectors):
        for j, q in enumerate(q_vecs):
            target = G + q
            diffs = G_vectors - target[np.newaxis, :]
            dist = np.linalg.norm(diffs, axis=1)
            m2 = np.argmin(dist)
            if dist[m2] < 1e-6:
                i1, i2 = 4*m, 4*m2
                H[i1:i1+2, i2+2:i2+4] = T_mats[j]
                H[i2+2:i2+4, i1:i1+2] = T_mats[j].conj().T
    return H

# High-symmetry k-path

def high_symmetry_path(num_k=60):
    Gpt = np.array([0, 0])
    Kpt = Gm * np.array([2/3, 0])
    Mpt = Gm * np.array([1/2, np.sqrt(3)/2])
    path = []
    for start, end in [(Gpt, Kpt), (Kpt, Mpt), (Mpt, Gpt)]:
        for t in np.linspace(0, 1, num_k, endpoint=False):
            path.append((1-t)*start + t*end)
    path.append(Gpt)
    return np.array(path)

k_path = high_symmetry_path()

# Compute band structure
bands = np.zeros((len(k_path), 4*N))
for i, kpt in enumerate(k_path):
    ev, _ = eigh(build_Hk(kpt, G_vectors))
    bands[i, :] = np.sort(np.real(ev))

# Plot bands
plt.figure(figsize=(8,6))
for n in range(8): plt.plot(bands[:, n], lw=1)
plt.xticks([0,60,120,180], ["Γ","K","M","Γ"])
plt.ylabel("Energy (eV)")
plt.title(f"TBG Bands (θ={theta_deg}°)")
plt.grid(True)
plt.show()

# -- Validation Functions --

def flat_band_bandwidth(bands):
    # bands[:,1] & bands[:,2] are two flat bands
    bw = np.max(bands[:,2] - bands[:,1])
    return bw

def gap_at_K(bands):
    # index 60 is start of K point segment
    idx_K = len(k_path)//3
    gap = bands[idx_K,3] - bands[idx_K,2]
    return gap

def particle_hole_check(bands):
    # check E_n + E_{-n-1} ~ 0
    errs = bands + bands[:,::-1]
    return np.max(np.abs(errs))


# -- Corrected Validation Functions --

def flat_band_indices(N):
    """Return the zero-based indices of the two magic-angle flat bands."""
    mid = 2*N
    return mid - 1, mid

def flat_band_bandwidth(bands):
    """
    Compute the bandwidth of the two central (flat) bands.
    bands: array of shape (Nk,4N) sorted in ascending energy.
    """
    N4 = bands.shape[1]
    N = N4 // 4
    i1, i2 = flat_band_indices(N)
    # Maximum minus minimum energy across the k-path
    bw = np.max(bands[:, i2] - bands[:, i1])
    return bw

def gap_at_K(bands):
    """
    Compute the direct gap between the flat bands (i2) and the next band (i2+1)
    at the K-point position in your k_path.
    """
    N4 = bands.shape[1]
    N = N4 // 4
    _, i2 = flat_band_indices(N)
    # K-point index is Nk/3 if your path is evenly split into thirds
    idx_K = len(k_path) // 3
    # Gap between band i2+1 (above the flat bands) and band i2
    gap = bands[idx_K, i2+1] - bands[idx_K, i2]
    return gap

def particle_hole_check(bands):
    """Check E_n + E_{-(n+1)} ~ 0 for perfect particle-hole symmetry."""
    Nk, N4 = bands.shape
    errs = bands + bands[:, ::-1]
    return np.max(np.abs(errs))

# -- Re-run validations --

bw   = flat_band_bandwidth(bands)
gapK = gap_at_K(bands)
symm = particle_hole_check(bands)

print(f"Flat-band (middle) bandwidth: {bw*1000:.2f} meV")
print(f"Gap above flat bands at K-point: {gapK*1000:.2f} meV")
print(f"Max particle-hole asymmetry: {symm*1000:.2f} meV")

# Basis convergence test for the flat-band bandwidth:
def basis_convergence(shells_list=[2,3,4]):
    results = {}
    for s in shells_list:
        Gs = generate_G_vectors(shells=s)
        kp = high_symmetry_path()
        b = np.zeros((len(kp), 4*len(Gs)))
        for i, kpt in enumerate(kp):
            b[i,:] = np.sort(np.real(eigh(build_Hk(kpt, Gs))[0]))
        results[s] = flat_band_bandwidth(b)
    return results

conv = basis_convergence([2,3,4])
print("Bandwidth vs basis shells (meV):",
      {s: conv[s]*1000 for s in conv})
