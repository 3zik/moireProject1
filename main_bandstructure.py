import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from k_path_generation import high_symmetry_path  # Make sure this file is in the same folder

# --- Physical constants and parameters ---
a = 2.46e-10             # graphene lattice constant [m]
v_F = 1e6                # Fermi velocity [m/s]
hbar = 6.582119569e-16   # Planck's constant over 2π [eV·s]

theta_deg = 1.1
theta = np.deg2rad(theta_deg)

w0 = 0.0784  # AA tunneling (eV)
w1 = 0.110   # AB tunneling (eV)

# Derived quantities
k_theta = (4*np.pi/(3*a)) * 2*np.sin(theta/2)  # moiré momentum scale [1/m]
E0 = hbar * v_F * k_theta                       # energy scale [eV]
print(f"Energy scale E0 = {E0:.3f} eV")

# Dimensionless tunneling
w0_dim = w0 / E0
w1_dim = w1 / E0

# Rotation matrix
def R(angle):
    c,s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],[s, c]])

# Dirac K-point magnitude and dimensionless q vectors
K_mag = 4*np.pi/(3*a)
q1 = (R(-theta/2) @ np.array([0, K_mag]) - R(+theta/2) @ np.array([0, K_mag])) / k_theta
q2 = (R(-theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q3 = (R(-theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q_vecs = [q1, q2, q3]

# Pauli matrices
sigma0 = np.eye(2)
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])

# Tunneling matrices T_j
phis = [np.arctan2(q[1], q[0]) for q in q_vecs]
T_mats = [w0_dim*sigma0 + w1_dim*(np.cos(phi)*sigma_x + np.sin(phi)*sigma_y) for phi in phis]

# Moiré reciprocal lattice magnitude and G vectors
L_m = a / (2 * np.sin(theta/2))                 # moiré period [m]
Gm_phys = 4 * np.pi / (np.sqrt(3) * L_m)        # physical moiré G magnitude [1/m]

def generate_G_vectors_physical(shells=3):
    b1_phys = Gm_phys * np.array([np.sqrt(3)/2, 1/2])
    b2_phys = Gm_phys * np.array([0, 1])
    Gs = []
    for i in range(-shells, shells+1):
        for j in range(-shells, shells+1):
            Gs.append(i*b1_phys + j*b2_phys)
    return np.array(Gs)

def build_Hk(k_vec, G_vectors):
    N = len(G_vectors)
    dim = 4*N
    H = np.zeros((dim, dim), dtype=complex)
    
    # intralayer Dirac terms
    for m, G in enumerate(G_vectors):
        idx = 4*m
        k_top = R(+theta/2) @ k_vec - G
        k_bot = R(-theta/2) @ k_vec - G
        H0_top = k_top[0]*sigma_x + k_top[1]*sigma_y
        H0_bot = k_bot[0]*sigma_x + k_bot[1]*sigma_y
        H[idx:idx+2, idx:idx+2]     = H0_top
        H[idx+2:idx+4, idx+2:idx+4] = H0_bot
    
    # interlayer tunneling
    for m, G in enumerate(G_vectors):
        for j, q in enumerate(q_vecs):
            target = G + q
            diffs = G_vectors - target[np.newaxis,:]
            dist = np.linalg.norm(diffs, axis=1)
            m2 = np.argmin(dist)
            if dist[m2] < 1e-6:
                i1, i2 = 4*m, 4*m2
                H[i1:i1+2,   i2+2:i2+4] = T_mats[j]
                H[i2+2:i2+4, i1:i1+2]   = T_mats[j].conj().T
    return H

def compute_bands_along_path(G_vectors, shells=3, num_k=60):
    k_path = high_symmetry_path(num_k=num_k)
    N = len(G_vectors)
    bands_dim = np.zeros((len(k_path), 4 * N))

    for i, kpt in enumerate(k_path):
        Hk = build_Hk(kpt, G_vectors)
        ev, _ = eigh(Hk)
        bands_dim[i, :] = np.sort(np.real(ev))

    return k_path, bands_dim

def plot_bands(bands, num_k=60, theta_deg=1.1, N=None):
    plt.figure(figsize=(8,6))
    for n in range(2*N-3, 2*N+1):
        plt.plot(bands[:, n], lw=1)
    plt.xticks([0, num_k, 2*num_k, 3*num_k], ["Γ", "K", "M", "Γ"])
    plt.ylabel("Energy (eV)")
    plt.title(f"TBG Bands (θ={theta_deg}°, shells=3)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    shells = 3
    G_vectors_phys = generate_G_vectors_physical(shells)
    G_vectors = G_vectors_phys / k_theta
    N = len(G_vectors)

    k_path, bands_dim = compute_bands_along_path(G_vectors, shells=shells, num_k=60)
    bands = E0 * bands_dim

    plot_bands(bands, num_k=60, theta_deg=theta_deg, N=N)

import numpy as np
import matplotlib.pyplot as plt

# Assume bands is your (num_k_points, total_bands) numpy array of energies (in eV)

def plot_central_bands(bands, num_bands=4, num_k=180):
    total_bands = bands.shape[1]
    center = total_bands // 2
    half_window = num_bands // 2
    start_idx = center - half_window
    end_idx = center + half_window

    print(f"Total bands: {total_bands}")
    print(f"Plotting bands indices from {start_idx} to {end_idx-1}")

    plt.figure(figsize=(10,6))
    for n in range(start_idx, end_idx):
        plt.plot(bands[:, n], label=f'Band {n}')
    plt.xticks([0, num_k, 2*num_k, 3*num_k], ["Γ", "K", "M", "Γ"])
    plt.ylabel("Energy (eV)")
    plt.title("Twisted Bilayer Graphene - Central Flat Bands")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all_bands_near_zero(bands, energy_window=0.5):
    # Find all bands with any energy within ±energy_window eV
    mask = np.any((bands >= -energy_window) & (bands <= energy_window), axis=0)
    selected_indices = np.where(mask)[0]

    print(f"Number of bands within ±{energy_window} eV: {len(selected_indices)}")
    print(f"Indices: {selected_indices}")

    plt.figure(figsize=(10,6))
    for n in selected_indices:
        plt.plot(bands[:, n], label=f'Band {n}')
    plt.xticks([0, bands.shape[0]//3, 2*bands.shape[0]//3, bands.shape[0]-1], ["Γ", "K", "M", "Γ"])
    plt.ylabel("Energy (eV)")
    plt.title(f"All Bands Within ±{energy_window} eV Window")
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage (insert after you compute 'bands'):

num_k_points = 180  # or however many you used in your k-path

plot_central_bands(bands, num_bands=4, num_k=num_k_points)
plot_all_bands_near_zero(bands, energy_window=0.5)
