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

## Getting Started

### Prerequisites

Install required Python packages:

```bash
pip install numpy scipy matplotlib
