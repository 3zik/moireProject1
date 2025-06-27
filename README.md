# Twisted Bilayer Graphene (TBG) Continuum Model Project

This project attempts to implement the foundational Bistritzer–MacDonald (BM) continuum model for twisted bilayer graphene (TBG) band structure calculations. The model captures low-energy moiré bands arising from the relative twist of two graphene sheets near the magic angle (~1.1°).

---

## References

Bistritzer, R., & MacDonald, A. H. (2011). Moiré bands in twisted double-layer graphene. PNAS, 108(30), 12233-12237. arXiv:1010.1365

Cao, Y., et al. (2018). Unconventional superconductivity in magic-angle graphene superlattices. Nature, 556(7699), 43-50.

### Requirements
```bash
numpy >= 1.19.0
matplotlib >= 3.3.0
scipy >= 1.5.0
```

### Prerequisites

Install required Python packages:

```bash
pip install numpy scipy matplotlib
```
## main.py Graphs/Info Generated

1. Full Band Structure along High-Symmetry k-Path
- Complete energy spectrum along Γ→K→M→Γ path
- All bands colored with viridis colormap
- High-symmetry points marked
2. Zoomed-in Flat Bands near Charge Neutrality
- Focus on the four central flat bands near charge neutrality
- Color-coded: red (valence), blue (conduction)
- Energy range optimized for flat band region
3. Density of States (DOS) vs Energy
- DOS vs energy with Gaussian broadening
- Charge neutrality point marked
- Van Hove singularities visible
4. Band Gap and Flat-Band Bandwidth vs Twist Angle
- Flat band bandwidth vs twist angle (0.8° - 1.5°)
- Band gap evolution
- Magic angle (1.08°) highlighted
5. Bandwidth and Gap Convergence vs Basis Size (Shells)
- Bandwidth & gap vs number of shells (basis size)
- Validates numerical convergence
- Shows when results are stable
6. Particle-Hole Symmetry Validation Plot
- Histogram comparing positive/negative energy distributions
- Validates fundamental symmetry of the model
- Shows quality of numerical implementation
7. 2D Color Map of Band Energy over the Full Moiré Brillouin Zone
- Contour plots of four flat bands in momentum space
- Full moiré Brillouin zone coverage
- Shows band topology and extrema
8. 3D Surface Plot of Band Energy in Momentum Space
- Four 3D surface plots of flat band energies
- Visualizes band curvature and dispersion
- Interactive-style visualization
9. (Optional) Real-Space Localization of Flat-Band Wavefunctions
- Real-space localization at Γ, K, M points
- Scatter plots showing amplitude in G-space
- Reveals spatial character of flat band states
10. Analysis Summary
- Complete parameter summary
- File descriptions
- Model validation metrics

# Twisted Bilayer Graphene - Bistritzer-MacDonald Continuum Model

This implementation computes the electronic band structure of twisted bilayer graphene (TBG) using the continuum model approach of Bistritzer and MacDonald. The model describes how two graphene sheets twisted by a small angle create a moiré superlattice with flat bands near the magic twist angle (~1.1°). This code provides a complete framework for studying the electronic properties that lead to correlated physics and superconductivity in TBG.

## Features

### Core Calculations
- **Moiré Superlattice Generation**: Automatic generation of moiré reciprocal lattice vectors up to specified number of shells
- **Hamiltonian Construction**: Full Bistritzer-MacDonald Hamiltonian including:
 - Intralayer terms with rotated Dirac cones for each graphene layer
 - Interlayer tunneling with AA and AB stacking configurations
 - Configurable coupling strengths and twist angles
- **Band Structure Computation**: Eigenvalue solving along high-symmetry k-paths (Γ → K → M → Γ)
- **Density of States (DOS)**: Gaussian-broadened DOS calculation with customizable energy ranges
- **2D Band Structure**: Full Brillouin zone mapping for momentum-space visualization
- **Wavefunction Analysis**: Computation and visualization of flat-band wavefunctions

### Analysis Tools
- **Convergence Testing**: Systematic analysis of results vs basis set size (number of shells)
- **Twist Angle Dependence**: Automated calculation of bandwidth and gap vs twist angle
- **Validation Metrics**: 
 - Flat band bandwidth quantification
 - Band gap measurements at high-symmetry points
 - Particle-hole symmetry verification
 - Energy scale validation
- **Magic Angle Detection**: Identification of twist angles with minimal flat band bandwidth

## Visualization & Output

### Standard Plots
- **Full Band Structure**: Complete electronic band structure along high-symmetry paths
- **Flat Band Focus**: Zoomed view highlighting the four central flat bands near charge neutrality
- **Density of States**: Energy-resolved DOS showing van Hove singularities
- **2D Band Maps**: Contour plots of band energies across the full Brillouin zone
- **3D Surface Plots**: Three-dimensional visualization of band dispersion
- **Wavefunction Localization**: Real-space and momentum-space wavefunction analysis

### Comprehensive Analysis Suite
The `generate_comprehensive_analysis()` function produces:
1. Full band structure plots
2. Zoomed flat band visualization
3. Density of states calculation
4. Twist angle dependence analysis
5. Convergence testing plots
6. Particle-hole symmetry validation
7. 2D momentum-space band maps
8. 3D surface plots of band energies
9. Wavefunction localization analysis
10. Detailed summary report

All outputs are saved as high-quality PNG files (300 DPI) with accompanying analysis summary.

## Physical Significance

### Moiré Physics
- **Magic Angle Behavior**: The code accurately reproduces the magic angle (~1.08°) where flat bands emerge
- **Flat Band Formation**: Demonstrates how interlayer coupling creates nearly dispersionless bands
- **Correlated Physics**: Provides the electronic structure foundation for understanding superconductivity and correlated insulator states

### Key Physical Quantities
- **Bandwidth**: Flat band bandwidth typically ~10-50 meV near magic angle
- **Energy Gaps**: Band gaps at high-symmetry points indicating topological properties
- **Wavefunction Localization**: Real-space localization leading to strong correlations
- **Particle-Hole Symmetry**: Validation of fundamental symmetries in the model

### Experimental Relevance
- **Transport Properties**: Band structure directly relates to electrical conductivity
- **Optical Spectroscopy**: DOS and band gaps connect to optical absorption experiments
- **STM/STS**: Local density of states relevant for scanning tunneling measurements
- **ARPES**: Momentum-resolved band structure for angle-resolved photoemission

## Usage Example

### Basic Band Structure Calculation

```python
# Import the module
from tbg_model import TwistedBilayerGraphene

# Initialize model at magic angle
tbg = TwistedBilayerGraphene(twist_angle_deg=1.08, shells=3)

# Generate k-point path and compute band structure
k_path, k_distances, labels, label_positions = tbg.generate_k_path(n_points_per_segment=100)
eigenvalues = tbg.compute_band_structure(k_path)

# Validate and analyze results
metrics = tbg.validate_results(eigenvalues, k_path)
tbg.print_validation_metrics(metrics)

# Plot band structure
tbg.plot_band_structure(k_distances, eigenvalues, labels, label_positions)
```

## Berry Curvature Analysis (`berry.py`)

The Berry curvature analysis module provides comprehensive tools for computing and visualizing the topological properties of twisted bilayer graphene flat bands. This module implements state-of-the-art methods for calculating Berry curvature, Berry connections, and Chern numbers - fundamental quantities that characterize the topology of quantum states.

### Features

#### Core Topological Calculations
- **Berry Curvature**: Computes the geometric curvature in momentum space using both:
 - Direct method (Fukui-Hatsugai-Suzuki plaquette approach)
 - Connection-based method (finite difference derivatives)
- **Berry Connections**: Vector potential in k-space from gauge-fixed wavefunctions
- **Chern Numbers**: Topological invariants from Berry curvature integration over the Brillouin zone
- **Gauge Fixing**: Multiple algorithms to ensure smooth wavefunction phases:
 - Smooth overlap method
 - Parallel transport method

 - The Berry curvature is defined as: Ω_n(k) = Im[⟨∂_kx u_n(k)| ∂_ky u_n(k)⟩] - Im[⟨∂_ky u_n(k)| ∂_kx u_n(k)⟩]
 - The Chern number is the integral: C_n = (1/2π) ∫∫_BZ Ω_n(k) d²k
 - These quantities are gauge-invariant and topologically protected, making them fundamental characterizations of quantum states.

#### Advanced Analysis Tools
- **2D K-point Sampling**: Dense momentum space grids with customizable resolution
- **Twist Angle Sweeps**: Systematic analysis of topology vs twist angle
- **Band Structure Integration**: Links topological properties to electronic structure
- **Convergence Analysis**: Tools to verify numerical accuracy

#### Visualization & Output
- **Berry Curvature Maps**: 2D contour plots showing curvature distribution
- **Chern Number Plots**: Bar charts and convergence analysis
- **Comprehensive Reports**: Automated generation of publication-ready figures and data
- **Export Capabilities**: Numerical data in `.npz` format for further analysis

### Physical Significance

The Berry curvature analysis reveals the topological character of TBG's flat bands:

- **Chern Numbers**: Integer topological charges that remain quantized despite perturbations
- **Berry Curvature Hotspots**: Regions of high curvature often correlate with strong electron interactions
- **Topology-Correlation Connection**: Links geometric properties to many-body physics
- **Magic Angle Physics**: Topological properties change dramatically near magic angles

### Usage Example

```python
from main import TwistedBilayerGraphene
from berry import BerryAnalysis
import numpy as np

# Initialize TBG model at magic angle
theta = np.deg2rad(1.05)  # degrees
tbg = TwistedBilayerGraphene(theta=theta)

# Set up Berry analysis with 50×50 k-point grid
berry = BerryAnalysis(tbg, n_kx=50, n_ky=50, k_range=1.2)

# Run complete analysis
results = berry.run_full_analysis(
   gauge_method='smooth',
   curvature_method='direct'
)

# Generate comprehensive report
berry.generate_comprehensive_report(results)

# Analyze topology vs twist angle
theta_range = np.linspace(0.8, 1.3, 10) * np.pi/180
angle_results = berry.analyze_topology_vs_twist_angle(theta_range)
berry.plot_topology_vs_angle(angle_results)
```

