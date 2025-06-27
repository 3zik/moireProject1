#!/usr/bin/env python3
"""
Bistritzer-MacDonald Continuum Model for Twisted Bilayer Graphene (TBG)

This implementation computes the electronic band structure of twisted bilayer graphene
using the continuum model approach of Bistritzer and MacDonald. The model describes
how two graphene sheets twisted by a small angle create a moiré superlattice with
flat bands near the magic twist angle (~1.1°).

Physical concepts:
- Moiré superlattice: Long-period pattern from twisted graphene layers
- Dirac cones: Linear energy dispersion in each graphene layer
- Interlayer tunneling: Coupling between layers creates flat bands
- Magic angle: Twist angle where bands become nearly flat

Author: Implementation based on Bistritzer & MacDonald, PNAS 2011
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore', category=np.ComplexWarning)

class TwistedBilayerGraphene:
    """
    Bistritzer-MacDonald continuum model for twisted bilayer graphene.
    
    This class implements the full continuum model including:
    - Moiré reciprocal lattice vector generation
    - Hamiltonian construction with intralayer and interlayer terms
    - Band structure calculation along high-symmetry paths
    - Validation and analysis tools
    """
    
    def __init__(self, twist_angle_deg: float = 1.08, shells: int = 3):
        """
        Initialize TBG model parameters.
        
        Args:
            twist_angle_deg: Twist angle in degrees (magic angle ≈ 1.08°)
            shells: Number of shells of G-vectors to include in basis
        """
        # Physical constants and parameters
        self.hbar_vf = 2.1354  # eV·Å (ℏv_F for graphene)
        self.theta = np.deg2rad(twist_angle_deg)  # Twist angle in radians
        self.shells = shells
        
        # Graphene lattice constant and derived quantities
        self.a_graphene = 2.46  # Å
        self.a_moire = self.a_graphene / (2 * np.sin(self.theta / 2))  # Moiré lattice constant
        self.k_theta = 4 * np.pi / (3 * self.a_graphene) * np.sin(self.theta / 2)  # Moiré momentum scale
        
        # Tunneling parameters (from ab initio or fitting to experiments)
        self.w_AA = 0.0797  # eV, AA stacking interlayer coupling
        self.w_AB = 0.0975  # eV, AB stacking interlayer coupling
        
        # Energy scale for dimensionless units
        self.E_scale = self.hbar_vf * self.k_theta  # Energy scale in eV
        
        # Generate moiré reciprocal lattice vectors
        self.G_vectors = self._generate_G_vectors()
        self.n_bands = 4 * len(self.G_vectors)  # 4 bands per G-vector (2 layers × 2 sublattices)
        
        print(f"TBG Model initialized:")
        print(f"  Twist angle: {twist_angle_deg:.3f}°")
        print(f"  Moiré lattice constant: {self.a_moire:.1f} Å")
        print(f"  Energy scale: {self.E_scale:.3f} eV")
        print(f"  Basis size: {len(self.G_vectors)} G-vectors, {self.n_bands} bands")
    
    def _generate_G_vectors(self) -> np.ndarray:
        """
        Generate moiré reciprocal lattice vectors up to specified number of shells.
        
        The moiré reciprocal lattice is triangular with basis vectors:
        b₁ = k_θ(√3, 1), b₂ = k_θ(√3, -1)
        
        Returns:
            Array of G-vectors in dimensionless units (normalized by k_θ)
        """
        # Moiré reciprocal lattice basis vectors (dimensionless)
        b1 = np.array([np.sqrt(3), 1])
        b2 = np.array([np.sqrt(3), -1])
        
        G_vectors = []
        
        # Generate G-vectors within specified number of shells
        for n1 in range(-self.shells, self.shells + 1):
            for n2 in range(-self.shells, self.shells + 1):
                # Skip if outside shell limit (use hexagonal distance)
                if abs(n1) + abs(n2) + abs(-n1 - n2) > 2 * self.shells:
                    continue
                    
                G = n1 * b1 + n2 * b2
                G_vectors.append(G)
        
        # Sort by magnitude for better numerical properties
        G_vectors = np.array(G_vectors)
        magnitudes = np.linalg.norm(G_vectors, axis=1)
        sorted_indices = np.argsort(magnitudes)
        
        return G_vectors[sorted_indices]
    
    def _rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate 2D rotation matrix."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])
    
    def _pauli_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pauli matrices for sublattice degree of freedom."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        return sigma_x, sigma_y, sigma_z
    
    def construct_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """
        Construct the full Bistritzer-MacDonald Hamiltonian matrix.
        
        The Hamiltonian includes:
        1. Intralayer terms: Rotated Dirac Hamiltonians for each layer
        2. Interlayer terms: Tunneling between layers with AA and AB coupling
        
        Args:
            k: Bloch wavevector in dimensionless units (k/k_θ)
            
        Returns:
            Complex Hamiltonian matrix of size (4×N_G, 4×N_G)
            where N_G is the number of G-vectors
        """
        n_G = len(self.G_vectors)
        H = np.zeros((4 * n_G, 4 * n_G), dtype=complex)
        
        sigma_x, sigma_y, sigma_z = self._pauli_matrices()
        
        # Rotation matrices for ±θ/2
        R_plus = self._rotation_matrix(self.theta / 2)
        R_minus = self._rotation_matrix(-self.theta / 2)
        
        # Fill Hamiltonian block by block
        for i, G_i in enumerate(self.G_vectors):
            for j, G_j in enumerate(self.G_vectors):
                
                # Intralayer terms (diagonal in layer index)
                if i == j:  # Diagonal in G-space
                    # Top layer (+θ/2 rotation)
                    k_plus = R_plus @ (k + G_i)
                    H_top = (k_plus[0] * sigma_x + k_plus[1] * sigma_y)
                    H[4*i:4*i+2, 4*j:4*j+2] = H_top
                    
                    # Bottom layer (-θ/2 rotation)
                    k_minus = R_minus @ (k + G_i)
                    H_bottom = (k_minus[0] * sigma_x + k_minus[1] * sigma_y)
                    H[4*i+2:4*i+4, 4*j+2:4*j+4] = H_bottom
                
                # Interlayer tunneling terms
                G_diff = G_j - G_i
                
                # Check if G_diff corresponds to tunneling vectors
                if np.allclose(G_diff, [0, 0], atol=1e-10):
                    # AA stacking (same sublattice)
                    T_AA = self.w_AA / self.E_scale * np.eye(2, dtype=complex)
                    H[4*i:4*i+2, 4*j+2:4*j+4] = T_AA  # Top to bottom
                    H[4*i+2:4*i+4, 4*j:4*j+2] = T_AA.conj().T  # Bottom to top
                
                # AB stacking terms (connecting different sublattices)
                # The three AB tunneling vectors in the moiré reciprocal lattice
                tunneling_vectors = [
                    np.array([np.sqrt(3), 1]),    # G₁
                    np.array([0, -2]),            # G₂  
                    np.array([-np.sqrt(3), 1])   # G₃
                ]
                
                for t_vec in tunneling_vectors:
                    if np.allclose(G_diff, t_vec, atol=1e-10):
                        # AB tunneling matrix (couples different sublattices)
                        T_AB = self.w_AB / self.E_scale * np.array([[0, 1], [0, 0]], dtype=complex)
                        H[4*i:4*i+2, 4*j+2:4*j+4] = T_AB  # Top to bottom
                        H[4*i+2:4*i+4, 4*j:4*j+2] = T_AB.conj().T  # Bottom to top
                    elif np.allclose(G_diff, -t_vec, atol=1e-10):
                        # Reverse direction
                        T_AB = self.w_AB / self.E_scale * np.array([[0, 0], [1, 0]], dtype=complex)
                        H[4*i:4*i+2, 4*j+2:4*j+4] = T_AB  # Top to bottom
                        H[4*i+2:4*i+4, 4*j:4*j+2] = T_AB.conj().T  # Bottom to top
        
        return H
    
    def generate_k_path(self, n_points_per_segment: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate high-symmetry k-point path in the moiré Brillouin zone.
        
        Path: Γ → K → M → Γ
        - Γ: (0, 0)
        - K: (2√3/3, 0) in dimensionless units
        - M: (√3/2, 1/2)
        
        Args:
            n_points_per_segment: Number of k-points per segment
            
        Returns:
            k_path: Array of k-points along path
            k_distances: Cumulative distances along path
            labels: High-symmetry point labels
        """
        # High-symmetry points in moiré BZ (dimensionless units)
        Gamma = np.array([0, 0])
        K = np.array([2*np.sqrt(3)/3, 0])
        M = np.array([np.sqrt(3)/2, 0.5])
        
        # Generate path segments
        segments = [
            (Gamma, K, 'Γ', 'K'),
            (K, M, 'K', 'M'), 
            (M, Gamma, 'M', 'Γ')
        ]
        
        k_path = []
        k_distances = []
        labels = ['Γ']
        label_positions = [0]
        
        total_distance = 0
        
        for start, end, start_label, end_label in segments:
            # Generate points along segment (excluding start to avoid duplication)
            t = np.linspace(0, 1, n_points_per_segment + 1)[1:]
            segment_points = start[None, :] + t[:, None] * (end - start)[None, :]
            
            # Calculate distances
            if len(k_path) == 0:
                k_path.append(start)
                k_distances.append(0)
            
            for point in segment_points:
                k_path.append(point)
                distance = np.linalg.norm(point - k_path[-2])
                total_distance += distance
                k_distances.append(total_distance)
            
            # Add label for end point
            labels.append(end_label)
            label_positions.append(total_distance)
        
        return np.array(k_path), np.array(k_distances), labels, label_positions
    
    def compute_band_structure(self, k_path: np.ndarray) -> np.ndarray:
        """
        Compute energy eigenvalues along k-point path.
        
        Args:
            k_path: Array of k-points in dimensionless units
            
        Returns:
            Array of energy eigenvalues in eV, shape (n_kpoints, n_bands)
        """
        n_k = len(k_path)
        eigenvalues = np.zeros((n_k, self.n_bands))
        
        print(f"Computing band structure for {n_k} k-points...")
        
        for i, k in enumerate(k_path):
            if i % max(1, n_k // 10) == 0:
                print(f"  Progress: {100*i/n_k:.0f}%")
            
            # Construct and diagonalize Hamiltonian
            H = self.construct_hamiltonian(k)
            eigvals = eigh(H, eigvals_only=True)
            
            # Convert to eV and sort
            eigenvalues[i] = np.sort(np.real(eigvals)) * self.E_scale
        
        print("  Complete!")
        return eigenvalues
    
    def plot_band_structure(self, k_distances: np.ndarray, eigenvalues: np.ndarray, 
                          labels: List[str], label_positions: List[float],
                          focus_flat_bands: bool = True) -> None:
        """
        Plot the computed band structure.
        
        Args:
            k_distances: Distances along k-path
            eigenvalues: Energy eigenvalues in eV
            labels: High-symmetry point labels
            label_positions: Positions of high-symmetry points
            focus_flat_bands: Whether to highlight the central flat bands
        """
        plt.figure(figsize=(10, 8))
        
        n_bands = eigenvalues.shape[1]
        
        if focus_flat_bands and n_bands >= 4:
            # Plot all bands in light gray
            for i in range(n_bands):
                plt.plot(k_distances, eigenvalues[:, i], 'lightgray', alpha=0.5, linewidth=0.5)
            
            # Highlight the four central flat bands
            central_start = n_bands // 2 - 2
            colors = ['red', 'blue', 'blue', 'red']
            linewidths = [2, 2, 2, 2]
            
            for i, (color, lw) in enumerate(zip(colors, linewidths)):
                band_idx = central_start + i
                plt.plot(k_distances, eigenvalues[:, band_idx], 
                        color=color, linewidth=lw, alpha=0.8)
        else:
            # Plot all bands with different colors
            colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
            for i in range(n_bands):
                plt.plot(k_distances, eigenvalues[:, i], color=colors[i], linewidth=1)
        
        # Add high-symmetry point labels
        for label, pos in zip(labels, label_positions):
            plt.axvline(x=pos, color='black', linestyle='--', alpha=0.5)
            plt.text(pos, plt.ylim()[1], label, ha='center', va='bottom', fontsize=12)
        
        # Formatting
        plt.xlabel('k-path', fontsize=12)
        plt.ylabel('Energy (eV)', fontsize=12)
        plt.title(f'TBG Band Structure (θ = {np.rad2deg(self.theta):.2f}°)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(k_distances[0], k_distances[-1])
        
        # Focus on flat band region if requested
        if focus_flat_bands:
            energy_range = np.max(eigenvalues) - np.min(eigenvalues)
            center_energy = 0  # Assume charge neutrality at E=0
            plt.ylim(center_energy - 0.3 * energy_range, center_energy + 0.3 * energy_range)
        
        plt.tight_layout()
        plt.show()
    
    def validate_results(self, eigenvalues: np.ndarray, k_path: np.ndarray) -> Dict[str, float]:
        """
        Validate the computed band structure and extract key metrics.
        
        Args:
            eigenvalues: Energy eigenvalues in eV
            k_path: k-point path
            
        Returns:
            Dictionary of validation metrics
        """
        n_bands = eigenvalues.shape[1]
        metrics = {}
        
        # Find flat bands (central 4 bands)
        if n_bands >= 4:
            central_start = n_bands // 2 - 2
            flat_bands = eigenvalues[:, central_start:central_start+4]
            
            # Flat band bandwidth (width of central bands)
            bandwidths = [np.max(flat_bands[:, i]) - np.min(flat_bands[:, i]) 
                         for i in range(4)]
            metrics['flat_band_bandwidth'] = np.mean(bandwidths)
            metrics['max_flat_band_width'] = np.max(bandwidths)
            
            # Gap at K-point (find K-point in path)
            # K-point should be at approximately 1/3 of the path
            k_point_idx = len(k_path) // 3
            k_energies = eigenvalues[k_point_idx]
            
            # Gap between flat bands and higher bands
            flat_max = np.max(flat_bands[k_point_idx])
            higher_min = np.min(eigenvalues[k_point_idx, central_start+4:])
            metrics['gap_above_flat_bands'] = higher_min - flat_max
            
            # Gap between lower bands and flat bands  
            lower_max = np.max(eigenvalues[k_point_idx, :central_start])
            flat_min = np.min(flat_bands[k_point_idx])
            metrics['gap_below_flat_bands'] = flat_min - lower_max
        
        # Check particle-hole symmetry
        all_energies = eigenvalues.flatten()
        positive_energies = all_energies[all_energies > 0]
        negative_energies = all_energies[all_energies < 0]
        
        if len(positive_energies) > 0 and len(negative_energies) > 0:
            # Compare distributions
            symmetry_error = np.abs(np.mean(positive_energies) + np.mean(negative_energies))
            metrics['particle_hole_symmetry_error'] = symmetry_error
        
        # Energy scale validation
        metrics['total_bandwidth'] = np.max(eigenvalues) - np.min(eigenvalues)
        metrics['energy_scale_eV'] = self.E_scale
        
        return metrics
    
    def print_validation_metrics(self, metrics: Dict[str, float]) -> None:
        """Print validation metrics in a formatted way."""
        print("\n" + "="*50)
        print("VALIDATION METRICS")
        print("="*50)
        
        if 'flat_band_bandwidth' in metrics:
            print(f"Flat band bandwidth (avg):     {metrics['flat_band_bandwidth']*1000:.1f} meV")
            print(f"Max flat band width:           {metrics['max_flat_band_width']*1000:.1f} meV")
        
        if 'gap_above_flat_bands' in metrics:
            print(f"Gap above flat bands (K):      {metrics['gap_above_flat_bands']*1000:.1f} meV")
            print(f"Gap below flat bands (K):      {metrics['gap_below_flat_bands']*1000:.1f} meV")
        
        if 'particle_hole_symmetry_error' in metrics:
            print(f"Particle-hole symmetry error:  {metrics['particle_hole_symmetry_error']*1000:.1f} meV")
        
        print(f"Total bandwidth:               {metrics['total_bandwidth']:.3f} eV")
        print(f"Energy scale:                  {metrics['energy_scale_eV']:.3f} eV")
        print("="*50)


def main():
    """
    Main function to run the TBG band structure calculation.
    """
    # Model parameters
    twist_angle = 1.08  # degrees (close to magic angle)
    shells = 3          # Number of G-vector shells
    n_points = 100      # K-points per segment
    
    print("Twisted Bilayer Graphene - Bistritzer-MacDonald Model")
    print("="*55)
    
    # Initialize model
    tbg = TwistedBilayerGraphene(twist_angle_deg=twist_angle, shells=shells)
    
    # Generate k-point path
    print("\nGenerating high-symmetry k-point path...")
    k_path, k_distances, labels, label_positions = tbg.generate_k_path(n_points)
    
    # Compute band structure
    eigenvalues = tbg.compute_band_structure(k_path)
    
    # Validate results
    metrics = tbg.validate_results(eigenvalues, k_path)
    tbg.print_validation_metrics(metrics)
    
    # Plot results
    print("\nGenerating band structure plot...")
    tbg.plot_band_structure(k_distances, eigenvalues, labels, label_positions)
    
    # Additional analysis
    print(f"\nModel completed successfully!")
    print(f"Computed {len(k_path)} k-points with {tbg.n_bands} bands each")
    
    return tbg, eigenvalues, k_path, k_distances


if __name__ == "__main__":
    # Run the calculation
    tbg_model, energies, k_points, k_dist = main()
    
    # Example of how to access results
    print(f"\nResults stored in variables:")
    print(f"  tbg_model: TBG model object")
    print(f"  energies: Band energies array {energies.shape}")
    print(f"  k_points: K-point path array {k_points.shape}")
    print(f"  k_dist: Distances along k-path {k_dist.shape}")