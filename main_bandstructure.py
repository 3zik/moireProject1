import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from k_path_generation import high_symmetry_path  # Your k-path generator function

# ... [Include all your BM model setup functions here or import them if modularized] ...

def compute_bands_along_path(G_vectors, shells=3, num_k=60):
    k_path = high_symmetry_path(num_k=num_k)
    N = len(G_vectors)
    bands_dim = np.zeros((len(k_path), 4 * N))

    for i, kpt in enumerate(k_path):
        Hk = build_Hk(kpt, G_vectors)  # your Hamiltonian builder
        ev, _ = eigh(Hk)
        bands_dim[i, :] = np.sort(np.real(ev))

    return k_path, bands_dim

def plot_bands(bands, num_k=60, theta_deg=1.1, N=None):
    plt.figure(figsize=(8,6))

    # plot the four central bands
    for n in range(2*N-3, 2*N+1):
        plt.plot(bands[:, n], lw=1)

    plt.xticks([0, num_k, 2*num_k, 3*num_k], ["Γ", "K", "M", "Γ"])
    plt.ylabel("Energy (eV)")
    plt.title(f"TBG Bands (θ={theta_deg}°, shells=3)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Setup your G_vectors and other params here
    shells = 3
    G_vectors = generate_G_vectors_physical(shells)
    G_vectors = G_vectors / k_theta
    N = len(G_vectors)

    k_path, bands_dim = compute_bands_along_path(G_vectors, shells=shells, num_k=60)
    bands = E0 * bands_dim

    plot_bands(bands, num_k=60, theta_deg=theta_deg, N=N)
