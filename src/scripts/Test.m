%% ---------------------------------------------------------------
%  Reproducible testbench: T-BT vs BT T-POD vs POD T-ERA vs ERA (timing, error, param count)
%  ---------------------------------------------------------------

clear; clc;

%% Problem sizes
n = 100;                 % state per slice
m = 5;                   
l = 5;                   
s = 9;                   % mode-3 (tubal) length

% ERA/POD block sizes 
T = 19; 
L = 19;

%% ---------- A,B,C for POD & ERA ----------
rng(1, 'twister');       % global seed
seedA = 100; seedB = 200; seedC = 300;
rng(seedA);   A_PE = (1/200)*randn(n,n,s);
rng(seedB);   B_PE = randn(n,m,s);
rng(seedC);   C_PE = randn(l,n,s);

%% Build unfolded (bcirc) full system 
As_PE = bcirc(A_PE);                 % (n*s) x (n*s)
Bs_PE = bcirc(B_PE);                 % (n*s) x (m*s)
Cs_PE = bcirc(C_PE);                 % (l*s) x (n*s)

sys_PE = ss(As_PE, Bs_PE, Cs_PE, 0, -1);   
sys_hinfnorm_PE = hinfnorm(sys_PE);

%% ---------- A,B,C for BT ONLY (different seeds) ----------
rng(111); A_BT = (1/200)*randn(n,n,s);
rng(222); B_BT = randn(n,m,s);
rng(333); C_BT = randn(l,n,s);

% full system for BT baseline
As_BT = bcirc(A_BT);  Bs_BT = bcirc(B_BT);  Cs_BT = bcirc(C_BT);
sys_BT_full  = ss(As_BT, Bs_BT, Cs_BT, 0, -1);
sys_hinfnorm_BT = hinfnorm(sys_BT_full);

%% Truncation levels (remove k states per slice; keep n-k per slice)
k_list = 55:5:90;
numK   = numel(k_list);

% Preallocate results:
Time_BT   = zeros(numK, 2);    % [T-BT_time, BT_time]
Re_BT     = zeros(numK, 2);    % [T-BT_relErr, BT_relErr]
Num_BT    = zeros(numK, 2);    % [T-BT_paramCount, BT_paramCount]

Time_POD   = zeros(numK, 2);    % [T-POD_time, POD_time]
Re_POD     = zeros(numK, 2);    % [T-POD_relErr, POD_relErr]
Num_POD    = zeros(numK, 2);    % [T-POD_paramCount, POD_paramCount]

Time_ERA   = zeros(numK, 2);    % [T-ERA_time, ERA_time]
Re_ERA     = zeros(numK, 2);    % [T-ERA_relErr, ERA_relErr]
Num_ERA    = zeros(numK, 2);    % [T-ERA_paramCount, ERA_paramCount]

%% Main loop over truncation k
for i = 1:numK
    k = k_list(i);

    %-------------------  T-BT and BT  -----------------
    % ---- T-BT on tensor (slice-wise) ----
    tStart = tic;
    [A_red_TBT, B_red_TBT, C_red_TBT] = T_BT(A_BT, B_BT, C_BT, k);
    Time_BT(i,1) = toc(tStart);

    % Build bcirc reduced system for T-BT 
    As_TBT = bcirc(A_red_TBT);
    Bs_TBT = bcirc(B_red_TBT);
    Cs_TBT = bcirc(C_red_TBT);
    sys_TBT = ss(As_TBT, Bs_TBT, Cs_TBT, 0, -1);

    % ---- BT on bcirc (matrix) model ----
    tStart = tic;
    [A_red_BT, B_red_BT, C_red_BT] = BT(A_BT, B_BT, C_BT, k);
    Time_BT(i,2) = toc(tStart);

    % BT returns bcirc (matrix) operators already
    sys_BT = ss(A_red_BT, B_red_BT, C_red_BT, 0, -1);

    % ---- Relative H-infinity errors ----
    Re_BT(i,1) = hinfnorm(sys_BT_full-sys_TBT)/sys_hinfnorm_BT;   % T-BT
    Re_BT(i,2) = hinfnorm(sys_BT_full-sys_BT)/sys_hinfnorm_BT;    % BT

    % ---- Parameter counts ----
    Num_BT(i,1) = numel(A_red_TBT) + numel(B_red_TBT) + numel(C_red_TBT);  % T-BT (tensor form)
    Num_BT(i,2) = numel(A_red_BT) + numel(B_red_BT) + numel(C_red_BT);     % BT (matrix/unfolded)
    
    
    
    %-------------------  T-POD and POD  -----------------
    % ---- T-POD on tensor (slice-wise) ----
    tStart = tic;
    [A_red_TPOD, B_red_TPOD, C_red_TPOD] = T_POD(A_PE, B_PE, C_PE, k, T, L);
    Time_POD(i,1) = toc(tStart);

    % Build bcirc reduced system for T-POD 
    As_TPOD = bcirc(A_red_TPOD);
    Bs_TPOD = bcirc(B_red_TPOD);
    Cs_TPOD = bcirc(C_red_TPOD);
    sys_TPOD = ss(As_TPOD, Bs_TPOD, Cs_TPOD, 0, -1);

    % ---- POD on bcirc (matrix) model ----
    tStart = tic;
    [A_red_POD, B_red_POD, C_red_POD] = POD(A_PE, B_PE, C_PE, k, T, L);
    Time_POD(i,2) = toc(tStart);

    % POD returns bcirc (matrix) operators already
    sys_POD = ss(A_red_POD, B_red_POD, C_red_POD, 0, -1);

    % ---- Relative H-infinity errors ----
    Re_POD(i,1) = hinfnorm(sys_PE-sys_TPOD)/sys_hinfnorm_PE;   % T-POD
    Re_POD(i,2) = hinfnorm(sys_PE-sys_POD)/sys_hinfnorm_PE;    % POD

    % ---- Parameter counts ----
    Num_POD(i,1) = numel(A_red_TPOD) + numel(B_red_TPOD) + numel(C_red_TPOD);  % T-POD (tensor form)
    Num_POD(i,2) = numel(A_red_POD) + numel(B_red_POD) + numel(C_red_POD);     % POD (matrix/unfolded)
    
    
    
    %-------------------  T-ERA and ERA  -----------------
    % ---- T-ERA on tensor (slice-wise) ----
    tStart = tic;
    [A_red_TERA, B_red_TERA, C_red_TERA] = T_ERA(A_PE, B_PE, C_PE, k, T, L);
    Time_ERA(i,1) = toc(tStart);

    % Build bcirc reduced system for T-ERA 
    As_TERA = bcirc(A_red_TERA);
    Bs_TERA = bcirc(B_red_TERA);
    Cs_TERA = bcirc(C_red_TERA);
    sys_TERA = ss(As_TERA, Bs_TERA, Cs_TERA, 0, -1);

    % ---- ERA on bcirc (matrix) model ----
    tStart = tic;
    [A_red_ERA, B_red_ERA, C_red_ERA] = ERA(A_PE, B_PE, C_PE, k, T, L);
    Time_ERA(i,2) = toc(tStart);

    % ERA returns bcirc (matrix) operators already
    sys_ERA = ss(A_red_ERA, B_red_ERA, C_red_ERA, 0, -1);

    % ---- Relative H-infinity errors ----
    Re_ERA(i,1) = hinfnorm(sys_PE-sys_TERA)/sys_hinfnorm_PE;   % T-ERA
    Re_ERA(i,2) = hinfnorm(sys_PE-sys_ERA)/sys_hinfnorm_PE;    % ERA

    % ---- Parameter counts ----
    Num_ERA(i,1) = numel(A_red_TERA) + numel(B_red_TERA) + numel(C_red_TERA);  % T-ERA (tensor form)
    Num_ERA(i,2) = numel(A_red_ERA) + numel(B_red_ERA) + numel(C_red_ERA);     % ERA (matrix/unfolded)
end