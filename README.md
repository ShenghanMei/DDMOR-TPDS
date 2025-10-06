# Data-Driven Model Order Reduction via T-SVD (Code)

This repository accompanies the paper **“Data-Driven Model Order Reduction via T-SVD”**  
by Shenghan Mei, Ziqin He, Yidan Mei, Xin Mao, Anqi Dong, Ren Wang, and Can Chen.

We provide T-product–based methods and their matrix-based counterparts:

- **T-Balanced Truncation (T-BT)** and **Balanced Truncation (BT)**
- **T-Balanced Proper Orthogonal Decomposition (T-BPOD)** and **Balanced Proper Orthogonal Decomposition (BPOD)**
- **T-Eigensystem Realization Algorithm (T-ERA)** and **Eigensystem Realization Algorithm (ERA)**

The repo includes reproducible experiments, including an MNIST-based case study.

---

## Requirements

- MATLAB
- [Tensor-Tensor Product Toolbox](https://github.com/canyilu/tensor-tensor-product-toolbox)

> Make sure the toolbox is on your MATLAB path before running experiments.

---

## Repository Structure

- `src/`
  - `algorithms/` — core methods (`T_BT.m`, `BT.m`, `T_POD.m`, `POD.m`, `T_ERA.m`, `ERA.m`)
  - `scripts/` — runnable demos
    - `test.m` — runs BT/T-BT, POD/T-POD, ERA/T-ERA; reports runtime, parameter counts, and relative errors
    - `scaling_TBT_BT.m` — scaling analysis for T-BT vs BT

- `experiments/`
  - `mnist/` — MNIST reconstruction example (`Digit1_Frames16.mat`, `Digit1.mat`, `mnist1_tera_era_reconstruction.m`)


---

