"""
Berry Curvature and Topological Analysis for Twisted Bilayer Graphene

This module computes the Berry curvature, Berry connections, and Chern numbers
for the flat bands of twisted bilayer graphene using the Bistritzer-MacDonald
continuum model. The implementation handles gauge fixing, numerical derivatives,
and topological invariant calculations.

Physical concepts:
- Berry curvature: Geometric curvature in momentum space
- Berry connection: Vector potential in k-space
- Chern numbers: Topological invariants from Berry curvature integration
- Gauge fixing: Ensuring smooth phase evolution of wavefunctions

Implementation features:
- Dense 2D k-point sampling over moiré Brillouin zone
- Numerical derivatives with finite differences
- Gauge fixing algorithms for smooth eigenvector phases
- Berry curvature calculation and visualization
- Chern number integration with error estimates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simpson
from typing import Tuple, List, Dict, Optional
import warnings
from tqdm import tqdm
import os

# Import the TBG model (assumes main.py is in same directory)
try:
    from main import TwistedBilayerGraphene
except ImportError:
    print("Error: Cannot import TwistedBilayerGraphene from main.py")
    print("Make sure main.py is in the same directory as berry.py")
    raise

class BerryAnalysis:
    """
    Comprehensive Berry curvature and topological analysis for TBG flat bands.
    
    This class provides methods to:
    1. Compute eigenvectors on dense 2D k-point grids
    2. Apply gauge fixing for smooth phase evolution
    3. Calculate Berry connections and curvatures
    4. Integrate Chern numbers and other topological invariants
    5. Visualize Berry curvature maps and topological properties
    """
    
    def __init__(self, tbg_model: TwistedBilayerGraphene, n_kx: int = 50, n_ky: int = 50,
                 k_range: float = 1.0):
        """
        Initialize Berry analysis for TBG model.
        
        Args:
            tbg_model: Initialized TwistedBilayerGraphene model
            n_kx, n_ky: Number of k-points in each direction
            k_range: Range of k-points in units of moiré reciprocal lattice vectors
        """
        self.tbg = tbg_model
        self.n_kx = n_kx
        self.n_ky = n_ky
        self.k_range = k_range
        
        # Generate 2D k-point grid
        self.kx = np.linspace(-k_range, k_range, n_kx)
        self.ky = np.linspace(-k_range, k_range, n_ky)
        self.kx_grid, self.ky_grid = np.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Grid spacing for derivatives
        self.dk_x = self.kx[1] - self.kx[0]
        self.dk_y = self.ky[1] - self.ky[0]
        
        # Storage for computed quantities
        self.eigenvalues = None
        self.eigenvectors = None
        self.berry_connection = None
        self.berry_curvature = None
        self.chern_numbers = None
        
        print(f"Berry Analysis initialized:")
        print(f"  K-point grid: {n_kx} × {n_ky} = {n_kx * n_ky} points")
        print(f"  K-range: ±{k_range} (dimensionless)")
        print(f"  Grid spacing: Δkx = {self.dk_x:.4f}, Δky = {self.dk_y:.4f}")
        print(f"  Total bands: {tbg_model.n_bands}")
    
    def compute_eigensystem_2d(self, band_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors on 2D k-point grid.
        
        Args:
            band_indices: List of band indices to keep (default: flat bands)
            
        Returns:
            eigenvalues: Shape (n_kx, n_ky, n_bands)
            eigenvectors: Shape (n_kx, n_ky, n_basis, n_bands)
        """
        if band_indices is None:
            # Default to flat bands (central 4 bands)
            n_total = self.tbg.n_bands
            band_indices = list(range(n_total//2 - 2, n_total//2 + 2))
        
        n_bands = len(band_indices)
        n_basis = self.tbg.n_bands  # Size of Hamiltonian matrix
        
        # Initialize arrays
        eigenvalues = np.zeros((self.n_kx, self.n_ky, n_bands))
        eigenvectors = np.zeros((self.n_kx, self.n_ky, n_basis, n_bands), dtype=complex)
        
        print(f"Computing eigensystem on {self.n_kx}×{self.n_ky} grid...")
        print(f"Keeping {n_bands} bands: {band_indices}")
        
        # Progress bar
        total_points = self.n_kx * self.n_ky
        with tqdm(total=total_points, desc="Computing eigensystem") as pbar:
            
            for i in range(self.n_kx):
                for j in range(self.n_ky):
                    k_point = np.array([self.kx_grid[i, j], self.ky_grid[i, j]])
                    
                    # Construct and diagonalize Hamiltonian
                    H = self.tbg.construct_hamiltonian(k_point)
                    eigvals, eigvecs = eigh(H)
                    
                    # Sort by eigenvalue and select desired bands
                    sorted_indices = np.argsort(np.real(eigvals))
                    
                    for band_idx, original_idx in enumerate(band_indices):
                        sorted_band_idx = sorted_indices[original_idx]
                        eigenvalues[i, j, band_idx] = np.real(eigvals[sorted_band_idx])
                        eigenvectors[i, j, :, band_idx] = eigvecs[:, sorted_band_idx]
                    
                    pbar.update(1)
        
        # Convert eigenvalues to eV
        eigenvalues *= self.tbg.E_scale
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        print("Eigensystem computation complete!")
        return eigenvalues, eigenvectors
    
    def fix_gauge_2d(self, eigenvectors: np.ndarray, method: str = 'smooth') -> np.ndarray:
        """
        Apply gauge fixing to ensure smooth evolution of eigenvector phases.
        
        The gauge freedom in quantum mechanics allows multiplication of eigenvectors
        by arbitrary phases. For Berry curvature calculations, we need smooth
        phase evolution across the Brillouin zone.
        
        Args:
            eigenvectors: Complex eigenvectors [n_kx, n_ky, n_basis, n_bands]
            method: Gauge fixing method ('smooth', 'parallel_transport')
            
        Returns:
            gauge_fixed_eigenvectors: Gauge-fixed eigenvectors
        """
        n_kx, n_ky, n_basis, n_bands = eigenvectors.shape
        fixed_eigenvectors = eigenvectors.copy()
        
        print(f"Applying gauge fixing using '{method}' method...")
        
        if method == 'smooth':
            # Smooth gauge: maximize overlap with previous k-point
            
            with tqdm(total=n_kx * n_ky, desc="Gauge fixing") as pbar:
                for i in range(n_kx):
                    for j in range(n_ky):
                        if i == 0 and j == 0:
                            # Reference point - no gauge fixing needed
                            pbar.update(1)
                            continue
                        
                        # Find reference eigenvector (previous k-point)
                        if j > 0:
                            ref_eigvecs = fixed_eigenvectors[i, j-1]  # Previous in y
                        else:
                            ref_eigvecs = fixed_eigenvectors[i-1, n_ky-1]  # Wrap from previous row
                        
                        current_eigvecs = eigenvectors[i, j]
                        
                        # For each band, find optimal phase to maximize overlap
                        for band in range(n_bands):
                            ref_vec = ref_eigvecs[:, band]
                            current_vec = current_eigvecs[:, band]
                            
                            # Compute overlap and optimal phase
                            overlap = np.vdot(ref_vec, current_vec)
                            optimal_phase = np.angle(overlap)
                            
                            # Apply phase correction
                            fixed_eigenvectors[i, j, :, band] = current_vec * np.exp(-1j * optimal_phase)
                        
                        pbar.update(1)
        
        elif method == 'parallel_transport':
            # Parallel transport gauge fixing
            with tqdm(total=n_kx * n_ky, desc="Parallel transport gauge fixing") as pbar:
                
                # Fix gauge along first row
                for j in range(1, n_ky):
                    ref_eigvecs = fixed_eigenvectors[0, j-1]
                    current_eigvecs = eigenvectors[0, j]
                    
                    # Compute unitary transformation that maximally aligns eigenvectors
                    overlap_matrix = np.conj(ref_eigvecs).T @ current_eigvecs
                    U, _, Vh = np.linalg.svd(overlap_matrix)
                    unitary_transform = U @ Vh
                    
                    fixed_eigenvectors[0, j] = current_eigvecs @ unitary_transform.T
                    pbar.update(1)
                
                # Fix gauge for remaining rows
                for i in range(1, n_kx):
                    for j in range(n_ky):
                        ref_eigvecs = fixed_eigenvectors[i-1, j]
                        current_eigvecs = eigenvectors[i, j]
                        
                        overlap_matrix = np.conj(ref_eigvecs).T @ current_eigvecs
                        U, _, Vh = np.linalg.svd(overlap_matrix)
                        unitary_transform = U @ Vh
                        
                        fixed_eigenvectors[i, j] = current_eigvecs @ unitary_transform.T
                        pbar.update(1)
        
        print("Gauge fixing complete!")
        return fixed_eigenvectors
    
    def compute_berry_connection(self, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Berry connection components using finite differences.
        
        The Berry connection is defined as:
        A_x^n(k) = i⟨u_n(k)|∂/∂k_x|u_n(k)⟩
        A_y^n(k) = i⟨u_n(k)|∂/∂k_y|u_n(k)⟩
        
        Args:
            eigenvectors: Gauge-fixed eigenvectors [n_kx, n_ky, n_basis, n_bands]
            
        Returns:
            A_x, A_y: Berry connection components [n_kx, n_ky, n_bands]
        """
        n_kx, n_ky, n_basis, n_bands = eigenvectors.shape
        A_x = np.zeros((n_kx, n_ky, n_bands), dtype=complex)
        A_y = np.zeros((n_kx, n_ky, n_bands), dtype=complex)
        
        print("Computing Berry connection using finite differences...")
        
        with tqdm(total=n_kx * n_ky, desc="Berry connection") as pbar:
            for i in range(n_kx):
                for j in range(n_ky):
                    
                    # Compute derivatives using finite differences with periodic boundary conditions
                    # ∂u/∂k_x
                    i_next = (i + 1) % n_kx
                    i_prev = (i - 1) % n_kx
                    du_dkx = (eigenvectors[i_next, j] - eigenvectors[i_prev, j]) / (2 * self.dk_x)
                    
                    # ∂u/∂k_y  
                    j_next = (j + 1) % n_ky
                    j_prev = (j - 1) % n_ky
                    du_dky = (eigenvectors[i, j_next] - eigenvectors[i, j_prev]) / (2 * self.dk_y)
                    
                    # Berry connection components
                    u_current = eigenvectors[i, j]
                    
                    for band in range(n_bands):
                        u_n = u_current[:, band]
                        du_dkx_n = du_dkx[:, band]
                        du_dky_n = du_dky[:, band]
                        
                        A_x[i, j, band] = 1j * np.vdot(u_n, du_dkx_n)
                        A_y[i, j, band] = 1j * np.vdot(u_n, du_dky_n)
                    
                    pbar.update(1)
        
        self.berry_connection = (A_x, A_y)
        print("Berry connection computation complete!")
        return A_x, A_y
    
    def compute_berry_curvature(self, A_x: np.ndarray, A_y: np.ndarray) -> np.ndarray:
        """
        Compute Berry curvature from Berry connection.
        
        The Berry curvature is:
        Ω_n(k) = ∂A_y^n/∂k_x - ∂A_x^n/∂k_y
        
        Args:
            A_x, A_y: Berry connection components [n_kx, n_ky, n_bands]
            
        Returns:
            berry_curvature: Berry curvature [n_kx, n_ky, n_bands]
        """
        n_kx, n_ky, n_bands = A_x.shape
        berry_curvature = np.zeros((n_kx, n_ky, n_bands))
        
        print("Computing Berry curvature...")
        
        with tqdm(total=n_kx * n_ky, desc="Berry curvature") as pbar:
            for i in range(n_kx):
                for j in range(n_ky):
                    
                    # Compute derivatives of Berry connection
                    # ∂A_y/∂k_x
                    i_next = (i + 1) % n_kx
                    i_prev = (i - 1) % n_kx
                    dAy_dkx = (A_y[i_next, j] - A_y[i_prev, j]) / (2 * self.dk_x)
                    
                    # ∂A_x/∂k_y - Fixed typo: n_vy -> n_ky
                    j_next = (j + 1) % n_ky
                    j_prev = (j - 1) % n_ky
                    dAx_dky = (A_x[i, j_next] - A_x[i, j_prev]) / (2 * self.dk_y)
                    
                    # Berry curvature
                    berry_curvature[i, j] = np.real(dAy_dkx - dAx_dky)
                    
                    pbar.update(1)
        
        self.berry_curvature = berry_curvature
        print("Berry curvature computation complete!")
        return berry_curvature
    
    def compute_berry_curvature_direct(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Compute Berry curvature directly from eigenvectors using the Fukui-Hatsugai-Suzuki method.
        
        This method computes Berry curvature as:
        Ω = Im[ln(U_1 * U_2 * U_3 * U_4)]
        
        where U_i are link variables around elementary plaquettes.
        
        Args:
            eigenvectors: Gauge-fixed eigenvectors [n_kx, n_ky, n_basis, n_bands]
            
        Returns:
            berry_curvature: Berry curvature [n_kx, n_ky, n_bands]
        """
        n_kx, n_ky, n_basis, n_bands = eigenvectors.shape
        berry_curvature = np.zeros((n_kx, n_ky, n_bands))
        
        print("Computing Berry curvature using Fukui-Hatsugai-Suzuki method...")
        
        with tqdm(total=n_kx * n_ky, desc="FHS Berry curvature") as pbar:
            for i in range(n_kx):
                for j in range(n_ky):
                    
                    # Define plaquette corners with periodic boundary conditions
                    i1, j1 = i, j
                    i2, j2 = (i + 1) % n_kx, j
                    i3, j3 = (i + 1) % n_kx, (j + 1) % n_ky
                    i4, j4 = i, (j + 1) % n_ky  # Fixed typo: n_vy -> n_ky
                    
                    # Get eigenvectors at plaquette corners
                    u1 = eigenvectors[i1, j1]  # (n_basis, n_bands)
                    u2 = eigenvectors[i2, j2]
                    u3 = eigenvectors[i3, j3]  
                    u4 = eigenvectors[i4, j4]
                    
                    # Compute link variables for each band
                    for band in range(n_bands):
                        # Link variables around plaquette
                        U12 = np.vdot(u1[:, band], u2[:, band])
                        U23 = np.vdot(u2[:, band], u3[:, band])
                        U34 = np.vdot(u3[:, band], u4[:, band])
                        U41 = np.vdot(u4[:, band], u1[:, band])
                        
                        # Berry curvature from plaquette
                        W = U12 * U23 * U34 * U41
                        # Handle potential numerical issues with log
                        if np.abs(W) > 1e-10:
                            berry_curvature[i, j, band] = np.imag(np.log(W))
                        else:
                            berry_curvature[i, j, band] = 0
                    
                    pbar.update(1)
        
        # Normalize by plaquette area
        berry_curvature /= (self.dk_x * self.dk_y)
        
        self.berry_curvature = berry_curvature
        print("Direct Berry curvature computation complete!")
        return berry_curvature
    
    def compute_chern_numbers(self, berry_curvature: np.ndarray) -> np.ndarray:
        """
        Compute Chern numbers by integrating Berry curvature over Brillouin zone.
        
        Chern number: C_n = (1/2π) ∫∫ Ω_n(k) dk_x dk_y
        
        Args:
            berry_curvature: Berry curvature [n_kx, n_ky, n_bands]
            
        Returns:
            chern_numbers: Chern number for each band
        """
        _, _, n_bands = berry_curvature.shape
        chern_numbers = np.zeros(n_bands)
        
        print("Computing Chern numbers...")
        
        for band in range(n_bands):
            # Integrate over 2D Brillouin zone using Simpson's rule
            omega_band = berry_curvature[:, :, band]
            
            # 2D integration
            integral = simpson([simpson(omega_band[i, :], x=self.ky) for i in range(self.n_kx)], 
                             x=self.kx)
            
            chern_numbers[band] = integral / (2 * np.pi)
        
        self.chern_numbers = chern_numbers
        
        print("Chern numbers:")
        for i, C in enumerate(chern_numbers):
            print(f"  Band {i+1}: C = {C:.6f}")
        
        return chern_numbers
    
    def run_full_analysis(self, band_indices: Optional[List[int]] = None, 
                         gauge_method: str = 'smooth',
                         curvature_method: str = 'direct') -> Dict:
        """
        Run complete Berry curvature analysis.
        
        Args:
            band_indices: Bands to analyze (default: flat bands)
            gauge_method: Gauge fixing method ('smooth' or 'parallel_transport')
            curvature_method: Berry curvature method ('direct' or 'connection')
            
        Returns:
            results: Dictionary containing all computed quantities
        """
        print("="*60)
        print("BERRY CURVATURE ANALYSIS")
        print("="*60)
        
        # 1. Compute eigensystem
        eigenvalues, eigenvectors = self.compute_eigensystem_2d(band_indices)
        
        # 2. Fix gauge
        fixed_eigenvectors = self.fix_gauge_2d(eigenvectors, method=gauge_method)
        
        # 3. Compute Berry curvature
        if curvature_method == 'direct':
            berry_curvature = self.compute_berry_curvature_direct(fixed_eigenvectors)
        else:  # connection method
            A_x, A_y = self.compute_berry_connection(fixed_eigenvectors)
            berry_curvature = self.compute_berry_curvature(A_x, A_y)
        
        # 4. Compute Chern numbers
        chern_numbers = self.compute_chern_numbers(berry_curvature)
        
        # 5. Store results
        results = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'fixed_eigenvectors': fixed_eigenvectors,
            'berry_curvature': berry_curvature,
            'chern_numbers': chern_numbers,
            'k_grid': (self.kx_grid, self.ky_grid)
        }
        
        if curvature_method == 'connection':
            results['berry_connection'] = self.berry_connection
        
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return results
    
    def plot_berry_curvature(self, berry_curvature: np.ndarray, 
                           band_indices: Optional[List[int]] = None,
                           save_plots: bool = True, output_dir: str = "berry_analysis"):
        """
        Plot Berry curvature maps for each band.
        
        Args:
            berry_curvature: Berry curvature data
            band_indices: Band indices for labeling
            save_plots: Whether to save plots
            output_dir: Directory for saved plots
        """
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        n_bands = berry_curvature.shape[2]
        
        if band_indices is None:
            band_indices = list(range(n_bands))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(min(n_bands, 4)):
            ax = axes[i]
            
            # Berry curvature map
            omega = berry_curvature[:, :, i]
            im = ax.contourf(self.kx_grid, self.ky_grid, omega, 
                           levels=20, cmap='RdBu_r', extend='both')
            
            # Add contour lines
            ax.contour(self.kx_grid, self.ky_grid, omega, 
                      levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            ax.set_xlabel('$k_x$ (dimensionless)', fontsize=10)
            ax.set_ylabel('$k_y$ (dimensionless)', fontsize=10)
            ax.set_title(f'Berry Curvature - Band {band_indices[i]+1}', fontsize=12)
            ax.set_aspect('equal')
            
            # Colorbar
            plt.colorbar(im, ax=ax, label='$\\Omega$ (dimensionless)')
            
            # Mark high-symmetry points
            ax.plot(0, 0, 'ko', markersize=5, label='$\\Gamma$')
            ax.plot(2*np.sqrt(3)/3, 0, 'ks', markersize=5, label='K')
            ax.plot(-2*np.sqrt(3)/3, 0, 'ks', markersize=5)
            ax.plot(np.sqrt(3)/2, 0.5, 'k^', markersize=5, label='M')
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/berry_curvature_maps.png", dpi=300, bbox_inches='tight')
            print(f"Berry curvature maps saved to {output_dir}/berry_curvature_maps.png")
        
        plt.show()
    
    def plot_chern_number_convergence(self, results: Dict, save_plots: bool = True, 
                                    output_dir: str = "berry_analysis"):
        """
        Plot convergence of Chern numbers with k-point density.
        
        Args:
            results: Results dictionary from full analysis
            save_plots: Whether to save plots
            output_dir: Directory for saved plots
        """
        # This would require running analysis at different k-point densities
        # For now, create a placeholder plot showing the computed values
        
        chern_numbers = results['chern_numbers']
        n_bands = len(chern_numbers)
        
        plt.figure(figsize=(10, 6))
        
        band_labels = [f'Band {i+1}' for i in range(n_bands)]
        colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
        
        plt.bar(band_labels, chern_numbers, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='C = ±1')
        plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        plt.xlabel('Band Index', fontsize=12)
        plt.ylabel('Chern Number', fontsize=12)
        plt.title(f'Chern Numbers for TBG Flat Bands (θ = {np.rad2deg(self.tbg.theta):.2f}°)', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text annotations
        for i, C in enumerate(chern_numbers):
            plt.text(i, C + 0.05 * np.sign(C) if C != 0 else 0.05, f'{C:.3f}', 
                    ha='center', va='bottom' if C >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/chern_numbers.png", dpi=300, bbox_inches='tight')
            print(f"Chern numbers plot saved to {output_dir}/chern_numbers.png")
        
        plt.show()
    
    def generate_comprehensive_report(self, results: Dict, 
                                    output_dir: str = "berry_analysis"):
        """
        Generate comprehensive analysis report with all plots and data.
        
        Args:
            results: Results dictionary from full analysis
            output_dir: Directory for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating comprehensive Berry analysis report...")
        print(f"Output directory: {output_dir}")
        
        # 1. Berry curvature maps
        self.plot_berry_curvature(results['berry_curvature'], 
                                 save_plots=True, output_dir=output_dir)
        
        # 2. Chern number plot
        self.plot_chern_number_convergence(results, save_plots=True, output_dir=output_dir)
        
        # 3. Band structure comparison plot
        eigenvalues = results['eigenvalues']
        
        plt.figure(figsize=(12, 8))
        n_bands = eigenvalues.shape[2]
        
        # Plot average band energies
        avg_energies = np.mean(eigenvalues, axis=(0, 1))
        band_indices = range(n_bands)
        
        plt.bar(band_indices, avg_energies, alpha=0.7, color='skyblue', 
               edgecolor='black', label='Average Energy')
        
        # Add Chern numbers as text
        chern_numbers = results['chern_numbers']
        for i, (E, C) in enumerate(zip(avg_energies, chern_numbers)):
            plt.text(i, E + 0.01, f'C={C:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.xlabel('Band Index', fontsize=12)
        plt.ylabel('Average Energy (eV)', fontsize=12)
        plt.title(f'TBG Band Energies and Chern Numbers (θ = {np.rad2deg(self.tbg.theta):.2f}°)', 
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/band_energies_chern.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Save numerical data
        np.savez(f"{output_dir}/berry_analysis_data.npz",
                 kx_grid=results['k_grid'][0],
                 ky_grid=results['k_grid'][1],
                 eigenvalues=results['eigenvalues'],
                 berry_curvature=results['berry_curvature'],
                 chern_numbers=results['chern_numbers'])
        
        # 5. Generate text report
        self._write_text_report(results, output_dir)
        
        print(f"\nComprehensive report generated in '{output_dir}/'")
        print("Files created:")
        print("  - berry_curvature_maps.png")
        print("  - chern_numbers.png") 
        print("  - band_energies_chern.png")
        print("  - berry_analysis_data.npz")
        print("  - analysis_report.txt")
    
    def _write_text_report(self, results: Dict, output_dir: str):
        """Write detailed text report of analysis results."""
        
        report_path = f"{output_dir}/analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BERRY CURVATURE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model parameters
            f.write("MODEL PARAMETERS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Twist angle (θ):        {np.rad2deg(self.tbg.theta):.4f}°\n")
            f.write(f"Moiré length scale:     {self.tbg.L_M:.2f} nm\n")
            f.write(f"Magic angle:            {np.rad2deg(self.tbg.theta_magic):.4f}°\n")
            f.write(f"Energy scale:           {self.tbg.E_scale*1000:.2f} meV\n")
            f.write(f"Total bands:            {self.tbg.n_bands}\n")
            f.write(f"K-point grid:           {self.n_kx} × {self.n_ky}\n")
            f.write(f"K-range:                ±{self.k_range}\n\n")
            
            # Chern numbers
            f.write("CHERN NUMBERS:\n")
            f.write("-"*40 + "\n")
            chern_numbers = results['chern_numbers']
            total_chern = np.sum(chern_numbers)
            
            for i, C in enumerate(chern_numbers):
                f.write(f"Band {i+1:2d}:  C = {C:8.4f}\n")
            
            f.write(f"\nTotal Chern number: {total_chern:.6f}\n")
            f.write(f"Charge neutrality satisfied: {abs(total_chern) < 1e-3}\n\n")
            
            # Band gap analysis
            eigenvalues = results['eigenvalues']
            avg_energies = np.mean(eigenvalues, axis=(0, 1))
            
            f.write("BAND STRUCTURE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            for i, E in enumerate(avg_energies):
                f.write(f"Band {i+1:2d}:  ⟨E⟩ = {E*1000:8.2f} meV\n")
            
            # Find gaps
            gaps = np.diff(avg_energies) * 1000  # Convert to meV
            f.write("\nBand gaps:\n")
            for i, gap in enumerate(gaps):
                f.write(f"Gap {i+1}-{i+2}: {gap:8.2f} meV\n")
            
            # Berry curvature statistics
            f.write("\nBERRY CURVATURE STATISTICS:\n")
            f.write("-"*40 + "\n")
            berry_curvature = results['berry_curvature']
            
            for i in range(berry_curvature.shape[2]):
                omega = berry_curvature[:, :, i]
                f.write(f"Band {i+1}:\n")
                f.write(f"  Maximum:     {np.max(omega):10.4f}\n")
                f.write(f"  Minimum:     {np.min(omega):10.4f}\n")
                f.write(f"  Mean:        {np.mean(omega):10.4f}\n")
                f.write(f"  Std dev:     {np.std(omega):10.4f}\n")
                f.write(f"  Chern (∫Ω):  {chern_numbers[i]:10.4f}\n\n")
        
        print(f"Detailed report written to {report_path}")

    def analyze_topology_vs_twist_angle(self, theta_range: np.ndarray, 
                                       n_kx: int = 30, n_ky: int = 30) -> Dict:
        """
        Analyze how topological properties change with twist angle.
        
        Args:
            theta_range: Array of twist angles to analyze
            n_kx, n_ky: K-point grid size for each calculation
            
        Returns:
            results: Dictionary with angle-dependent results
        """
        print("Analyzing topology vs twist angle...")
        print(f"Twist angles: {np.rad2deg(theta_range)} degrees")
        
        # Store results
        angle_results = {
            'twist_angles': theta_range,
            'chern_numbers': [],
            'band_gaps': [],
            'berry_curvature_max': []
        }
        
        # Temporarily store original grid parameters
        orig_n_kx, orig_n_ky = self.n_kx, self.n_ky
        orig_kx, orig_ky = self.kx, self.ky
        orig_grids = self.kx_grid, self.ky_grid
        orig_dk = self.dk_x, self.dk_y
        
        try:
            # Update grid parameters for faster calculation
            self.n_kx, self.n_ky = n_kx, n_ky
            self.kx = np.linspace(-self.k_range, self.k_range, n_kx)
            self.ky = np.linspace(-self.k_range, self.k_range, n_ky)
            self.kx_grid, self.ky_grid = np.meshgrid(self.kx, self.ky, indexing='ij')
            self.dk_x = self.kx[1] - self.kx[0]
            self.dk_y = self.ky[1] - self.ky[0]
            
            for theta in tqdm(theta_range, desc="Twist angle sweep"):
                # Update TBG model with new angle
                self.tbg.theta = theta
                self.tbg._compute_parameters()
                
                # Run analysis for this angle
                eigenvalues, eigenvectors = self.compute_eigensystem_2d()
                fixed_eigenvectors = self.fix_gauge_2d(eigenvectors, method='smooth')
                berry_curvature = self.compute_berry_curvature_direct(fixed_eigenvectors)
                chern_numbers = self.compute_chern_numbers(berry_curvature)
                
                # Extract relevant quantities
                avg_energies = np.mean(eigenvalues, axis=(0, 1))
                gaps = np.diff(avg_energies)
                max_berry_curvature = [np.max(np.abs(berry_curvature[:, :, i])) 
                                     for i in range(berry_curvature.shape[2])]
                
                # Store results
                angle_results['chern_numbers'].append(chern_numbers)
                angle_results['band_gaps'].append(gaps)
                angle_results['berry_curvature_max'].append(max_berry_curvature)
        
        finally:
            # Restore original grid parameters
            self.n_kx, self.n_ky = orig_n_kx, orig_n_ky
            self.kx, self.ky = orig_kx, orig_ky
            self.kx_grid, self.ky_grid = orig_grids
            self.dk_x, self.dk_y = orig_dk
        
        # Convert lists to arrays
        angle_results['chern_numbers'] = np.array(angle_results['chern_numbers'])
        angle_results['band_gaps'] = np.array(angle_results['band_gaps'])
        angle_results['berry_curvature_max'] = np.array(angle_results['berry_curvature_max'])
        
        return angle_results
    
    def plot_topology_vs_angle(self, angle_results: Dict, save_plots: bool = True,
                              output_dir: str = "berry_analysis"):
        """Plot topological properties vs twist angle."""
        
        theta_deg = np.rad2deg(angle_results['twist_angles'])
        chern_numbers = angle_results['chern_numbers']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Chern numbers vs angle
        ax1 = axes[0, 0]
        n_bands = chern_numbers.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
        
        for i in range(n_bands):
            ax1.plot(theta_deg, chern_numbers[:, i], 'o-', 
                    color=colors[i], label=f'Band {i+1}', linewidth=2)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Twist Angle (degrees)')
        ax1.set_ylabel('Chern Number')
        ax1.set_title('Chern Numbers vs Twist Angle')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Total Chern number
        ax2 = axes[0, 1]
        total_chern = np.sum(chern_numbers, axis=1)
        ax2.plot(theta_deg, total_chern, 'ko-', linewidth=2, markersize=6)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Twist Angle (degrees)')
        ax2.set_ylabel('Total Chern Number')
        ax2.set_title('Charge Neutrality Check')
        ax2.grid(True, alpha=0.3)
        
        # 3. Band gaps
        ax3 = axes[1, 0]
        band_gaps = angle_results['band_gaps'] * 1000  # Convert to meV
        n_gaps = band_gaps.shape[1]
        
        for i in range(n_gaps):
            ax3.plot(theta_deg, band_gaps[:, i], 'o-', 
                    label=f'Gap {i+1}-{i+2}', linewidth=2)
        
        ax3.set_xlabel('Twist Angle (degrees)')
        ax3.set_ylabel('Band Gap (meV)')
        ax3.set_title('Band Gaps vs Twist Angle')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Maximum Berry curvature
        ax4 = axes[1, 1]
        max_berry = angle_results['berry_curvature_max']
        
        for i in range(n_bands):
            ax4.plot(theta_deg, max_berry[:, i], 'o-', 
                    color=colors[i], label=f'Band {i+1}', linewidth=2)
        
        ax4.set_xlabel('Twist Angle (degrees)')
        ax4.set_ylabel('Max |Berry Curvature|')
        ax4.set_title('Berry Curvature Strength vs Angle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/topology_vs_angle.png", dpi=300, bbox_inches='tight')
            print(f"Topology vs angle plot saved to {output_dir}/topology_vs_angle.png")
        
        plt.show()


def run_berry_analysis_example():
    """
    Example script showing how to use the Berry analysis tools.
    """
    print("Berry Curvature Analysis Example")
    print("="*50)
    
    # 1. Initialize TBG model
    try:
        from main import TwistedBilayerGraphene
        
        # Create TBG model near magic angle
        theta_degrees = 1.05  # Close to first magic angle
        tbg = TwistedBilayerGraphene(theta=np.deg2rad(theta_degrees))
        print(f"TBG model initialized at θ = {theta_degrees}°")
        
    except ImportError:
        print("Error: Could not import TwistedBilayerGraphene from main.py")
        print("Make sure main.py is in the same directory.")
        return
    
    # 2. Initialize Berry analysis
    # Use moderate grid size for example (increase for production)
    berry_analysis = BerryAnalysis(tbg, n_kx=40, n_ky=40, k_range=1.2)
    
    # 3. Run full analysis
    results = berry_analysis.run_full_analysis(
        gauge_method='smooth',
        curvature_method='direct'
    )
    
    # 4. Generate comprehensive report
    berry_analysis.generate_comprehensive_report(results)
    
    # 5. Optional: Analyze topology vs twist angle
    print("\nRunning twist angle analysis...")
    
    # Define range of twist angles around magic angle
    theta_magic_deg = np.rad2deg(tbg.theta_magic)
    theta_range_deg = np.linspace(theta_magic_deg - 0.2, theta_magic_deg + 0.2, 5)
    theta_range_rad = np.deg2rad(theta_range_deg)
    
    angle_results = berry_analysis.analyze_topology_vs_twist_angle(
        theta_range_rad, n_kx=25, n_ky=25
    )
    
    berry_analysis.plot_topology_vs_angle(angle_results)
    
    print("\nBerry analysis complete!")
    print("Check the 'berry_analysis/' directory for all output files.")


if __name__ == "__main__":
    # Run example analysis
    run_berry_analysis_example()