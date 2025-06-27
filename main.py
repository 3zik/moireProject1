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

Implementation based on Bistritzer & MacDonald, PNAS 2011
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from typing import Tuple, List, Dict
import warnings

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


    def compute_dos(self, eigenvalues: np.ndarray, energy_range: Tuple[float, float] = None,
                   n_points: int = 1000, broadening: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute density of states using Gaussian broadening.
        
        Args:
            eigenvalues: Energy eigenvalues in eV
            energy_range: Energy range for DOS calculation
            n_points: Number of energy points
            broadening: Gaussian broadening in eV
            
        Returns:
            energies: Energy points
            dos: Density of states
        """
        if energy_range is None:
            energy_range = (np.min(eigenvalues) - 0.1, np.max(eigenvalues) + 0.1)
        
        energies = np.linspace(energy_range[0], energy_range[1], n_points)
        dos = np.zeros(n_points)
        
        # Gaussian broadening
        for eigen_vals in eigenvalues:
            for E in eigen_vals:
                dos += np.exp(-(energies - E)**2 / (2 * broadening**2))
        
        # Normalize
        dos /= (np.sqrt(2 * np.pi) * broadening * len(eigenvalues))
        
        return energies, dos
    
    def compute_2d_band_structure(self, n_kx: int = 50, n_ky: int = 50, 
                                 energy_range: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute band structure over 2D grid in Brillouin zone.
        
        Args:
            n_kx, n_ky: Number of k-points in each direction
            energy_range: Range of k-points (in units of reciprocal lattice vectors)
            
        Returns:
            kx_grid, ky_grid: 2D k-point grids
            band_energies: 3D array of band energies [kx, ky, band]
        """
        # Create k-point grid
        kx = np.linspace(-energy_range, energy_range, n_kx)
        ky = np.linspace(-energy_range, energy_range, n_ky)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        band_energies = np.zeros((n_kx, n_ky, self.n_bands))
        
        print(f"Computing 2D band structure on {n_kx}×{n_ky} grid...")
        
        for i in range(n_kx):
            if i % max(1, n_kx // 10) == 0:
                print(f"  Progress: {100*i/n_kx:.0f}%")
            
            for j in range(n_ky):
                k_point = np.array([kx_grid[i, j], ky_grid[i, j]])
                H = self.construct_hamiltonian(k_point)
                eigvals = eigh(H, eigvals_only=True)
                band_energies[i, j] = np.sort(np.real(eigvals)) * self.E_scale
        
        print("  Complete!")
        return kx_grid, ky_grid, band_energies
    
    def compute_wavefunctions(self, k_points: np.ndarray, band_indices: List[int] = None) -> np.ndarray:
        """
        Compute wavefunctions for specified bands and k-points.
        
        Args:
            k_points: Array of k-points
            band_indices: List of band indices to compute (default: flat bands)
            
        Returns:
            wavefunctions: Complex wavefunctions [k_point, band, G_vector, sublattice]
        """
        if band_indices is None:
            # Default to flat bands
            n_bands = self.n_bands
            band_indices = list(range(n_bands//2 - 2, n_bands//2 + 2))
        
        n_k = len(k_points)
        n_bands_compute = len(band_indices)
        n_G = len(self.G_vectors)
        
        wavefunctions = np.zeros((n_k, n_bands_compute, n_G, 4), dtype=complex)
        
        for i, k in enumerate(k_points):
            H = self.construct_hamiltonian(k)
            eigvals, eigvecs = eigh(H)
            
            # Sort by eigenvalue
            sorted_indices = np.argsort(np.real(eigvals))
            
            for j, band_idx in enumerate(band_indices):
                wf = eigvecs[:, sorted_indices[band_idx]]
                # Reshape to [G_vector, sublattice]
                wavefunctions[i, j] = wf.reshape(n_G, 4)
        
        return wavefunctions

def generate_comprehensive_analysis(tbg: TwistedBilayerGraphene, 
                                  eigenvalues: np.ndarray, 
                                  k_path: np.ndarray, 
                                  k_distances: np.ndarray,
                                  labels: List[str], 
                                  label_positions: List[float]):
    """
    Generate comprehensive analysis plots and save them as files.
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create output directory
    output_dir = "tbg_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating comprehensive analysis plots...")
    print(f"Output directory: {output_dir}")
    
    # 1. Full Band Structure along High-Symmetry k-Path
    print("  1. Full band structure...")
    plt.figure(figsize=(12, 8))
    n_bands = eigenvalues.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
    
    for i in range(n_bands):
        plt.plot(k_distances, eigenvalues[:, i], color=colors[i], linewidth=1)
    
    for label, pos in zip(labels, label_positions):
        plt.axvline(x=pos, color='black', linestyle='--', alpha=0.5)
        plt.text(pos, plt.ylim()[1]*0.95, label, ha='center', va='top', fontsize=12)
    
    plt.xlabel('k-path', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'TBG Full Band Structure (θ = {np.rad2deg(tbg.theta):.2f}°)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_full_band_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Zoomed-in Flat Bands near Charge Neutrality
    print("  2. Zoomed flat bands...")
    plt.figure(figsize=(12, 8))
    
    # Plot all bands in light gray
    for i in range(n_bands):
        plt.plot(k_distances, eigenvalues[:, i], 'lightgray', alpha=0.3, linewidth=0.5)
    
    # Highlight flat bands
    central_start = n_bands // 2 - 2
    colors = ['red', 'blue', 'blue', 'red']
    for i, color in enumerate(colors):
        band_idx = central_start + i
        plt.plot(k_distances, eigenvalues[:, band_idx], 
                color=color, linewidth=2, alpha=0.8, label=f'Band {band_idx+1}')
    
    for label, pos in zip(labels, label_positions):
        plt.axvline(x=pos, color='black', linestyle='--', alpha=0.5)
        plt.text(pos, plt.ylim()[1]*0.95, label, ha='center', va='top', fontsize=12)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('k-path', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'TBG Flat Bands (θ = {np.rad2deg(tbg.theta):.2f}°)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom to flat band region
    flat_bands = eigenvalues[:, central_start:central_start+4]
    energy_center = np.mean(flat_bands)
    energy_span = np.max(flat_bands) - np.min(flat_bands)
    plt.ylim(energy_center - 2*energy_span, energy_center + 2*energy_span)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_flat_bands_zoom.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Density of States (DOS) vs Energy
    print("  3. Density of states...")
    energies, dos = tbg.compute_dos(eigenvalues, broadening=0.005)
    
    plt.figure(figsize=(10, 8))
    plt.plot(energies, dos, 'b-', linewidth=2)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Charge neutrality')
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Density of States (states/eV)', fontsize=12)
    plt.title(f'TBG Density of States (θ = {np.rad2deg(tbg.theta):.2f}°)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Zoom to interesting region around charge neutrality
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_density_of_states.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Band Gap and Flat-Band Bandwidth vs Twist Angle
    print("  4. Angle dependence...")
    angles = np.linspace(0.8, 1.5, 15)
    bandwidths = []
    gaps = []
    
    for angle in angles:
        tbg_temp = TwistedBilayerGraphene(twist_angle_deg=angle, shells=2)  # Smaller basis for speed
        k_temp, _, _, _ = tbg_temp.generate_k_path(n_points_per_segment=30)
        eigs_temp = tbg_temp.compute_band_structure(k_temp)
        
        # Compute bandwidth and gap
        n_temp = eigs_temp.shape[1]
        flat_start = n_temp // 2 - 2
        flat_bands_temp = eigs_temp[:, flat_start:flat_start+4]
        bandwidth = np.max(flat_bands_temp) - np.min(flat_bands_temp)
        bandwidths.append(bandwidth * 1000)  # Convert to meV
        
        # Gap at K point (approximate)
        k_idx = len(k_temp) // 3
        gap = np.min(eigs_temp[k_idx, flat_start+4:]) - np.max(eigs_temp[k_idx, flat_start-1:flat_start+4])
        gaps.append(gap * 1000)  # Convert to meV
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(angles, bandwidths, 'ro-', linewidth=2, markersize=6)
    ax1.axvline(x=1.08, color='blue', linestyle='--', alpha=0.7, label='Magic angle')
    ax1.set_xlabel('Twist Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Flat Band Bandwidth (meV)', fontsize=12)
    ax1.set_title('Flat Band Bandwidth vs Twist Angle', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(angles, gaps, 'bo-', linewidth=2, markersize=6)
    ax2.axvline(x=1.08, color='blue', linestyle='--', alpha=0.7, label='Magic angle')
    ax2.set_xlabel('Twist Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Band Gap (meV)', fontsize=12)
    ax2.set_title('Band Gap vs Twist Angle', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_angle_dependence.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Bandwidth and Gap Convergence vs Basis Size (Shells)
    print("  5. Convergence analysis...")
    shell_range = range(1, 6)
    conv_bandwidths = []
    conv_gaps = []
    
    for shells in shell_range:
        tbg_temp = TwistedBilayerGraphene(twist_angle_deg=1.08, shells=shells)
        k_temp, _, _, _ = tbg_temp.generate_k_path(n_points_per_segment=30)
        eigs_temp = tbg_temp.compute_band_structure(k_temp)
        
        n_temp = eigs_temp.shape[1]
        flat_start = n_temp // 2 - 2
        flat_bands_temp = eigs_temp[:, flat_start:flat_start+4]
        bandwidth = np.max(flat_bands_temp) - np.min(flat_bands_temp)
        conv_bandwidths.append(bandwidth * 1000)
        
        k_idx = len(k_temp) // 3
        gap = np.min(eigs_temp[k_idx, flat_start+4:]) - np.max(eigs_temp[k_idx, flat_start-1:flat_start+4])
        conv_gaps.append(gap * 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(shell_range, conv_bandwidths, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Shells', fontsize=12)
    ax1.set_ylabel('Flat Band Bandwidth (meV)', fontsize=12)
    ax1.set_title('Convergence: Bandwidth vs Basis Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(shell_range, conv_gaps, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Shells', fontsize=12)
    ax2.set_ylabel('Band Gap (meV)', fontsize=12)
    ax2.set_title('Convergence: Gap vs Basis Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Particle-Hole Symmetry Validation Plot
    print("  6. Particle-hole symmetry...")
    plt.figure(figsize=(12, 8))
    
    # Plot E vs -E
    all_energies = eigenvalues.flatten()
    positive_energies = all_energies[all_energies > 0]
    negative_energies = all_energies[all_energies < 0]
    
    # Create histogram
    bins = np.linspace(-np.max(np.abs(all_energies)), np.max(np.abs(all_energies)), 50)
    hist_pos, _ = np.histogram(positive_energies, bins=bins)
    hist_neg, _ = np.histogram(-negative_energies, bins=bins)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    plt.plot(bin_centers, hist_pos, 'r-', linewidth=2, label='Positive energies')
    plt.plot(bin_centers, hist_neg, 'b--', linewidth=2, label='Negative energies (flipped)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Particle-Hole Symmetry Check', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/6_particle_hole_symmetry.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 2D Color Map of Band Energy over the Full Moiré Brillouin Zone
    print("  7. 2D band structure (this may take a while)...")
    kx_grid, ky_grid, band_energies_2d = tbg.compute_2d_band_structure(n_kx=30, n_ky=30)
    
    # Plot flat bands in 2D
    central_start = band_energies_2d.shape[2] // 2 - 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(4):
        band_idx = central_start + i
        im = axes[i].contourf(kx_grid, ky_grid, band_energies_2d[:, :, band_idx], 
                             levels=20, cmap='RdBu_r')
        axes[i].set_title(f'Flat Band {i+1}', fontsize=12)
        axes[i].set_xlabel('kx', fontsize=10)
        axes[i].set_ylabel('ky', fontsize=10)
        plt.colorbar(im, ax=axes[i], label='Energy (eV)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/7_2d_band_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 3D Surface Plot of Band Energy in Momentum Space
    print("  8. 3D surface plot...")
    fig = plt.figure(figsize=(15, 12))
    
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        band_idx = central_start + i
        
        surf = ax.plot_surface(kx_grid, ky_grid, band_energies_2d[:, :, band_idx],
                              cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('kx', fontsize=10)
        ax.set_ylabel('ky', fontsize=10)
        ax.set_zlabel('Energy (eV)', fontsize=10)
        ax.set_title(f'3D: Flat Band {i+1}', fontsize=12)
        
        # Add colorbar
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/8_3d_surface_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Real-Space Localization of Flat-Band Wavefunctions
    print("  9. Wavefunction localization...")
    
    # Select a few high-symmetry k-points
    k_points_special = np.array([[0, 0],  # Gamma
                                [2*np.sqrt(3)/3, 0],  # K
                                [np.sqrt(3)/2, 0.5]])  # M
    
    wavefunctions = tbg.compute_wavefunctions(k_points_special)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    k_labels = ['Γ', 'K', 'M']
    
    for k_idx, k_label in enumerate(k_labels):
        for band_idx in range(4):
            ax = axes[k_idx, band_idx]
            
            # Plot wavefunction amplitude in G-space
            wf = wavefunctions[k_idx, band_idx]
            amplitudes = np.sum(np.abs(wf)**2, axis=1)  # Sum over sublattices
            
            # Create scatter plot of amplitudes vs G-vector positions
            G_x = tbg.G_vectors[:, 0]
            G_y = tbg.G_vectors[:, 1]
            
            scatter = ax.scatter(G_x, G_y, c=amplitudes, s=100*amplitudes/np.max(amplitudes),
                               cmap='viridis', alpha=0.7)
            
            ax.set_title(f'{k_label}-point, Band {band_idx+1}', fontsize=10)
            ax.set_xlabel('Gx', fontsize=9)
            ax.set_ylabel('Gy', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/9_wavefunction_localization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary report
    print("  10. Summary report...")
    with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
        f.write("Twisted Bilayer Graphene - Bistritzer-MacDonald Model Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  Twist angle: {np.rad2deg(tbg.theta):.3f}°\n")
        f.write(f"  Number of shells: {tbg.shells}\n")
        f.write(f"  Basis size: {len(tbg.G_vectors)} G-vectors\n")
        f.write(f"  Total bands: {tbg.n_bands}\n")
        f.write(f"  Energy scale: {tbg.E_scale:.3f} eV\n")
        f.write(f"  Moiré lattice constant: {tbg.a_moire:.1f} Å\n\n")
        
        f.write("Generated Files:\n")
        f.write("  1. 1_full_band_structure.png - Complete band structure\n")
        f.write("  2. 2_flat_bands_zoom.png - Zoomed flat bands\n")
        f.write("  3. 3_density_of_states.png - DOS calculation\n")
        f.write("  4. 4_angle_dependence.png - Twist angle dependence\n")
        f.write("  5. 5_convergence_analysis.png - Basis size convergence\n")
        f.write("  6. 6_particle_hole_symmetry.png - Symmetry validation\n")
        f.write("  7. 7_2d_band_structure.png - 2D momentum space\n")
        f.write("  8. 8_3d_surface_plot.png - 3D surface plots\n")
        f.write("  9. 9_wavefunction_localization.png - Wavefunction analysis\n")
        f.write("  10. analysis_summary.txt - This summary file\n")
    
    print(f"\nAnalysis complete! All files saved to '{output_dir}/'")
    print("Generated 9 plots + 1 summary file")

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
    
    # Generate comprehensive analysis
    generate_comprehensive_analysis(tbg, eigenvalues, k_path, k_distances, labels, label_positions)
    
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