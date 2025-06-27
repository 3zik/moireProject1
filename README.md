# Twisted Bilayer Graphene (TBG) Continuum Model Project

This project implements the foundational Bistritzer–MacDonald (BM) continuum model for twisted bilayer graphene (TBG) band structure calculations. The model captures low-energy moiré bands arising from the relative twist of two graphene sheets near the magic angle (~1.1°).

---

## Repository Structure

- **k_path_generation/**  
  Generates the high-symmetry path through the moiré Brillouin zone (BZ) used for band structure calculations.  
  - `k_path_generation.py` — computes and plots the smooth k-path (Γ → K → M → Γ).

- **hamiltonian_evaluation/**  
  Builds the BM Hamiltonian matrix at a single k-point and computes eigenvalues (energy levels).  
  - `hamiltonian_evaluation.py` — constructs and diagonalizes the Hamiltonian at Γ or any k-point.

- **bandstructure_plotting/**  
  Computes the full band structure along the high-symmetry k-path and plots the resulting moiré band dispersion.  
  - `bandstructure_plotting.py` — calculates and plots bands along Γ-K-M-Γ.

---

### Prerequisites

Install required Python packages:

```bash
pip install numpy scipy matplotlib


## References

Bistritzer, R., & MacDonald, A. H. (2011). Moiré bands in twisted double-layer graphene. PNAS, 108(30), 12233-12237. arXiv:1010.1365

Cao, Y., et al. (2018). Unconventional superconductivity in magic-angle graphene superlattices. Nature, 556(7699), 43-50.