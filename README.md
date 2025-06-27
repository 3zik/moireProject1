# Twisted Bilayer Graphene (TBG) Continuum Model Project

This project implements the foundational Bistritzer–MacDonald (BM) continuum model for twisted bilayer graphene (TBG) band structure calculations. The model captures low-energy moiré bands arising from the relative twist of two graphene sheets near the magic angle (~1.1°).

---

### Prerequisites

Install required Python packages:

```bash
pip install numpy scipy matplotlib
```

## Graphs/Info Generated

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

## References

Bistritzer, R., & MacDonald, A. H. (2011). Moiré bands in twisted double-layer graphene. PNAS, 108(30), 12233-12237. arXiv:1010.1365

Cao, Y., et al. (2018). Unconventional superconductivity in magic-angle graphene superlattices. Nature, 556(7699), 43-50.