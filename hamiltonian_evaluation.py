# hamiltonian_evaluation.py

import numpy as np
from scipy.linalg import eigh

# Constants and parameters (same as your project)
a = 2.46e-10
v_F = 1e6
hbar = 6.582119569e-16
theta_deg = 1.1
theta = np.deg2rad(theta_deg)
w0 = 0.0784
w1 = 0.11

k_theta = (4*np.pi/(3*a)) * 2*np.sin(theta/2)
E0 = hbar * v_F * k_theta

w0_dim = w0 / E0
w1_dim = w1 / E0

def R(angle):
    c,s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],[s, c]])

# Pauli matrices
sigma0 = np.eye(2)
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])

K_mag = 4*np.pi/(3*a)
q1 = (R(-theta/2) @ np.array([0, K_mag]) - R(+theta/2) @ np.array([0, K_mag])) / k_theta
q2 = (R(-theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q3 = (R(-theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2]) - R(+theta/2) @ np.array([-K_mag*np.sqrt(3)/2, -K_mag/2])) / k_theta
q_vecs = [q1, q2, q3]

phis = [np.arctan2(q[1], q[0]) for q in q_vecs]
T_mats = [w0_dim*sigma0 + w1_dim*(np.cos(phi)*sigma_x + np.sin(phi)*sigma_y) for phi in phis]

# Generate G-vectors (small basis for test)
def generate_G_vectors(shells=1):
    Gm = 1.0  # dimensionless scale
    b1 = Gm * np.array([np.sqrt(3)/2, 1/2])
    b2 = Gm * np.array([0, 1])
    Gs = []
    for i in range(-shells, shells+1):
        for j in range(-shells, shells+1):
            Gs.append(i*b1 + j*b2)
    return np.array(Gs)

G_vectors = generate_G_vectors(shells=1)
N = len(G_vectors)

def build_Hk(k_vec, G_vectors):
    N = len(G_vectors)
    dim = 4*N
    H = np.zeros((dim, dim), dtype=complex)
    for m, G in enumerate(G_vectors):
        idx = 4*m
        k_top = R(+theta/2) @ (k_vec - G)
        k_bot = R(-theta/2) @ (k_vec - G)
        H0_top = k_top[0]*sigma_x + k_top[1]*sigma_y
        H0_bot = k_bot[0]*sigma_x + k_bot[1]*sigma_y
        H[idx:idx+2, idx:idx+2] = H0_top
        H[idx+2:idx+4, idx+2:idx+4] = H0_bot
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

if __name__ == "__main__":
    # Test at Γ point
    k_test = np.array([0.0, 0.0])
    H = build_Hk(k_test, G_vectors)
    eigvals, eigvecs = eigh(H)
    eigvals = np.real(eigvals) * E0  # convert to eV
    print("Eigenvalues at Γ (eV):")
    for val in eigvals:
        print(f"{val:.4f}")
