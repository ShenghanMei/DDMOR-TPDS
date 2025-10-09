# src

Source code for T-product–based model reduction and matrix baselines.

## Layout
- `algorithms/` — reusable functions:
  - `T_BT.m`, `BT.m`, `T_POD.m`, `POD.m`, `T_ERA.m`, `ERA.m`
- `scripts/` — runnable demos:
  - `Test.m` — runs T-BT/BT, T-POD/POD, T-ERA/ERA; reports runtimes, parameter counts and relative errors
  

## Usage
In MATLAB:
```matlab
addpath(genpath('src'));        % expose algorithms & scripts
run('src/scripts/test.m');      % main demo


