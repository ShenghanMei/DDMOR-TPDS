
clear; clc;

%% Sizes
n = 100;          % states per slice
m = 5;            
l = 5;            
k = 0;            % states removed per slice (so T-BT keeps n-k)

s_list = 2.^(0:10);                 % s = 1,2,4,...,1024
numS   = numel(s_list);

%% Preallocate results: [T-BT, BT]
Time_BT = nan(numS, 2);             % runtime seconds
Num_BT  = nan(numS, 2);             % parameter counts

%% Threshold for skipping BT (to avoid OOM)
BT_SKIP_S = 2^8;                     % skip BT for s >= 256

for i = 1:numS
    s = s_list(i);

    % Random instance
    A = (1/2000)*randn(n,n,s);
    B = randn(n,m,s);
    C = randn(l,n,s);

    % ---- T-BT ----
    tStart = tic;
    [A_red_TBT, B_red_TBT, C_red_TBT] = T_BT(A, B, C, k);
    Time_BT(i,1) = toc(tStart);

    % Parameter count (tensor form)
    Num_BT(i,1) = numel(A_red_TBT) + numel(B_red_TBT) + numel(C_red_TBT);

    % ---- BT (unfolded) ----
    if s < BT_SKIP_S
        % Try; if it still blows up or errors, record NaN and continue.
        try
            tStart = tic;
            [A_red_BT, B_red_BT, C_red_BT] = BT(A, B, C, k);
            Time_BT(i,2) = toc(tStart);

            Num_BT(i,2) = numel(A_red_BT) + numel(B_red_BT) + numel(C_red_BT);
        catch ME
            warning('BT failed for s=%d: %s', s, ME.message);
            % estimate BT parameter count analytically:
            Nr = (n - k) * s;           % reduced state dimension (matrix BT)
            Nu =  m * s;                % unfolded inputs
            Ny =  l * s;                % unfolded outputs
            Num_BT(i,2) = Nr^2 + Nr*Nu + Ny*Nr;   % = s^2*(n-k)[(n-k)+m+l]
            % Time_BT(i,2) remains NaN
        end
    else
        % Skip BT by design for large s; fill analytic param count if you want
        Nr = (n - k) * s;
        Nu =  m * s;
        Ny =  l * s;
        Num_BT(i,2) = Nr^2 + Nr*Nu + Ny*Nr;       % estimated params for BT
        % Time_BT(i,2) stays NaN to indicate "skipped"
    end
end

%% display quick summary
disp(table(s_list(:), Time_BT(:,1), Time_BT(:,2), Num_BT(:,1), Num_BT(:,2), ...
    'VariableNames', {'s','Time_TBT','Time_BT','Params_TBT','Params_BT'}));