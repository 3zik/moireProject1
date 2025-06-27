# k_path_generation.py

import numpy as np
import matplotlib.pyplot as plt

def high_symmetry_path(num_k=60):
    # Moiré reciprocal lattice vectors (dimensionless)
    b1 = np.array([np.sqrt(3)/2, 1/2])
    b2 = np.array([0, 1])

    # High-symmetry points in moiré BZ
    Γ = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2 * b1 + b2) / 3

    path = []
    for start, end in [(Γ, K), (K, M), (M, Γ)]:
        for t in np.linspace(0, 1, num_k, endpoint=False):
            path.append((1 - t) * start + t * end)
    path.append(Γ)
    return np.array(path)

if __name__ == "__main__":
    k_path = high_symmetry_path(num_k=60)
    print(f"Number of k-points: {len(k_path)}")

    plt.figure(figsize=(6,6))
    plt.plot(k_path[:,0], k_path[:,1], 'b.-')
    plt.title("High-Symmetry k-path in Moiré Brillouin Zone")
    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$k_y$")
    plt.grid(True)
    plt.axis('equal')

    # Mark special points
    special_points = {
        'Γ': 0,
        'K': 60,
        'M': 120,
        'Γ': 180
    }
    for label, idx in special_points.items():
        plt.plot(k_path[idx,0], k_path[idx,1], 'ro')
        plt.text(k_path[idx,0]+0.02, k_path[idx,1], label, fontsize=12)
    plt.show()
