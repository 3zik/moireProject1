import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# --- Physical constants and parameters ---
a = 2.46e-10             # graphene lattice constant (meters)
vF = 1e6                 # Fermi velocity (m/s)
hbar = 6.582119569e-16   # reduced Planck constant (eV·s)

theta_deg = 1.1          # twist angle in degrees (magic angle ~1.1°)
theta = np.deg2rad(theta_deg)

wAA = 0.0784             # AA interlayer tunneling amplitude (eV)
wAB = 0.110              # AB/BA interlayer tunneling amplitude (eV)

# --- Derived moiré quantities ---
k_theta = (4*np.pi/(3*a)) * 2*np.sin(theta/2)  # moiré momentum scale (1/m)
E0 = hbar * vF * k_theta                        # energy scale (eV)

print(f"Derived energy scale E0 = {E0:.3f} eV")

# Dimensionless tunneling amplitudes
w0 = wAA / E0
w1 = wAB / E0

# Rotation matrix helper
def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

# Generate moiré reciprocal lattice G-vectors up to given shell cutoff
def generate_G_vectors(shells=3):
    Lm = a / (2 * np.sin(theta/2))  # moiré lattice constant (m)
    Gm = 4*np.pi / (np.sqrt(3)*Lm)  # magnitude of moiré reciprocal vectors

    b1_phys = Gm * np.array([np.sqrt(3)/2, 0.5])
    b2_phys = Gm * np.array([0, 1])

    Gs = []
    for i in range(-shells, shells+1):
        for j in range(-shells, shells+1):
            Gs.append(i*b1_phys + j*b2_phys)
    Gs = np.array(Gs)

    # Normalize dimensionless G-vectors by k_theta
    return Gs / k_theta

# High-symmetry k-path in moiré BZ: Γ → K → M → Γ
def high_symmetry_path(num_k=60):
    b1 = np.array([np.sqrt(3)/2, 0.5])
    b2 = np.array([0, 1])

    Gamma = np.array([0, 0])
    K = (2*b1 + b2) / 3
    M = 0.5*b1

    path = []
    segments = [(Gamma, K), (K, M), (M, Gamma)]
    for start, end in segments:
        for t in np.linspace(0, 1, num_k, endpoint=False):
            path.append((1-t)*start + t*end)
    path.append(Gamma)  # close the loop
    return np.array(path)

# Pauli matrices
sigma0 = np.eye(2)
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])

# Dirac K-point magnitude in physical units
K_mag = 4*np.pi/(3*a)

# q vectors connecting valleys, dimensionless (using rotation matrices)
def calculate_q_vectors():
    q1 = (rotation(-theta/2) @ np.array([0, K_mag]) - rotation(theta/2) @ np.array([0, K_mag])) / k_theta
    q2 = (rotation(-theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2]) - rotation(theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
    q3 = (rotation(-theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2]) - rotation(theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
    return [q1, q2, q3]

q_vecs = calculate_q_vectors()

# Tunneling matrices T_j dimensionless
def tunneling_matrices():
    phis = [np.arctan2(q[1], q[0]) for q in q_vecs]
    T_list = [w0*sigma0 + w1*(np.cos(phi)*sigma_x + np.sin(phi)*sigma_y) for phi in phis]
    return T_list

T_mats = tunneling_matrices()

# Build Hamiltonian matrix H(k) for given k vector and G-vectors
def build_Hk(k_vec, G_vectors):
    N = len(G_vectors)
    dim = 4*N
    H = np.zeros((dim, dim), dtype=complex)

    for m, G in enumerate(G_vectors):
        idx = 4*m

        # Intralayer rotated Dirac cones
        k_top = rotation(+theta/2) @ k_vec - G
        k_bot = rotation(-theta/2) @ k_vec - G

        H0_top = k_top[0]*sigma_x + k_top[1]*sigma_y
        H0_bot = k_bot[0]*sigma_x + k_bot[1]*sigma_y

        H[idx:idx+2, idx:idx+2] = H0_top
        H[idx+2:idx+4, idx+2:idx+4] = H0_bot

    # Interlayer tunneling coupling
    for m, G in enumerate(G_vectors):
        for j, q in enumerate(q_vecs):
            target = G + q
            diffs = G_vectors - target[np.newaxis,:]
            dist = np.linalg.norm(diffs, axis=1)
            m2 = np.argmin(dist)
            if dist[m2] < 1e-6:
                i1, i2 = 4*m, 4*m2
                H[i1:i1+2, i2+2:i2+4] = T_mats[j]
                H[i2+2:i2+4, i1:i1+2] = T_mats[j].conj().T

    return H

# Compute bands along k-path
def compute_bands(G_vectors, num_k=60):
    k_path = high_symmetry_path(num_k)
    N = len(G_vectors)
    bands_dim = np.zeros((len(k_path), 4*N))

    for i, kpt in enumerate(k_path):
        Hk = build_Hk(kpt, G_vectors)
        ev, _ = eigh(Hk)
        bands_dim[i,:] = np.sort(np.real(ev))

    return k_path, bands_dim

# Plot the four central flat bands
def plot_bands(bands, num_k, theta_deg, N):
    plt.figure(figsize=(8,6))

    for n in range(2*N-3, 2*N+1):
        plt.plot(bands[:, n], lw=1)

    plt.xticks([0, num_k, 2*num_k, 3*num_k], ["Γ", "K", "M", "Γ"])
    plt.ylabel("Energy (eV)")
    plt.title(f"TBG Flat Bands, θ={theta_deg}°")
    plt.grid(True)
    plt.show()

# Validation metrics
def flat_band_indices(N):
    mid = 2*N
    return mid-1, mid

def flat_band_bandwidth(bands, N):
    i1, i2 = flat_band_indices(N)
    return np.max(bands[:, i2] - bands[:, i1])

def gap_above_flat_bands(bands, N):
    i1, i2 = flat_band_indices(N)
    idx_K = len(bands)//3
    return bands[idx_K, i2+1] - bands[idx_K, i2]

def particle_hole_symmetry_error(bands):
    # Check particle-hole symmetry by E(k) + E(-k) ~ 0
    # Here simplified by adding bands to reversed bands
    errs = bands + bands[:, ::-1]
    return np.max(np.abs(errs))

if __name__ == "__main__":
    shells = 3
    G_vectors = generate_G_vectors(shells)
    print(f"Dimensionless basis size N = {len(G_vectors)}")

    k_path, bands_dim = compute_bands(G_vectors, num_k=60)
    bands = E0 * bands_dim  # convert to physical units (eV)

    plot_bands(bands, num_k=60, theta_deg=theta_deg, N=len(G_vectors))

    bw = flat_band_bandwidth(bands, len(G_vectors))
    gap = gap_above_flat_bands(bands, len(G_vectors))
    ph_err = particle_hole_symmetry_error(bands)

    print(f"Flat-band bandwidth: {bw*1e3:.1f} meV")
    print(f"Gap above flat bands at K: {gap*1e3:.1f} meV")
    print(f"Particle-hole symmetry error: {ph_err*1e3:.1f} meV")
