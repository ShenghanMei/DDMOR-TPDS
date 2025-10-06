# src

Source code for T-product–based model reduction and matrix baselines.

## Layout
- `algorithms/` — reusable functions:
  - `T_BT.m`, `BT.m`, `T_POD.m`, `POD.m`, `T_ERA.m`, `ERA.m`
- `scripts/` — runnable demos:
  - `Test.m` — runs T-BT/BT, T-POD/POD, T-ERA/ERA; reports runtimes, parameter counts and relative errors
  - `scaling_TBT_BT.m` — scaling study vs. mode-3 size

## Usage
In MATLAB:
```matlab
addpath(genpath('src'));        % expose algorithms & scripts
run('src/scripts/test.m');      % main demo
% run('src/scripts/scaling_TBT_BT.m');

