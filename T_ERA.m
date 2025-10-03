function [A_red, B_red, C_red] = T_ERA(A, B, C, k, T, L)
% T_ERA  T-product–based ERA (T-ERA).
%
%   [A_red, B_red, C_red] = T_ERA(A, B, C, k, T, L)
%
% INPUTS
%   A : n x n x s   state transition tensor   (mode-3 length = s)
%   B : n x m x s   
%   C : l x n x s   
%   k : truncation parameter
%   T : number of block columns  minus 1  (so T+1 columns in Hankel)
%   L : number of block rows     minus 1  (so L+1 rows in Hankel)
%
% OUTPUTS (tensor form; t-product algebra)
%   A_red : r x r x s       
%   B_red : r x m x s
%   C_red : l x r x s
%
% METHOD
%   Build block Hankel tensors H and H_hat (one-step shifted); compute the
%   truncated t-SVD H ≈ U ∘ S ∘ V^T; form S^{-1/2}; then
%       A_red = S^{-1/2} ∘ U^T ∘ H1 ∘ V ∘ S^{-1/2}
%       B_red = S^{-1/2} ∘ U^T ∘ H(:, first block column)
%       C_red = (first block row of H) ∘ V ∘ S^{-1/2}
%
% -------------------------------------------------------------------------

    % -----------------------
    % Validate dimensions
    % -----------------------
    [n, n2, s] = size(A);
    if n ~= n2, error('A must be square on modes 1–2.'); end
    if size(B,1) ~= n || size(B,3) ~= s
        error('B must be n x m x s with the same n,s as A.'); end
    if size(C,2) ~= n || size(C,3) ~= s
        error('C must be l x n x s with the same n,s as A.'); end

    m = size(B,2);
    l = size(C,1);

    % Hankel block sizes
    nRows = l*(L+1);     % total rows in H
    nCols = m*(T+1);     % total cols in H

    r_keep_rows = nRows - k;   % rows kept in U,S
    r_keep_cols = nCols - k;   % cols kept in V,S
    
    r = min(r_keep_rows,r_keep_cols);

    % -----------------------
    % Build Z_i = C ∘ A^i ∘ B
    % and assemble block Hankel H, H1 (one-step shifted).
    % H has (L+1) block rows, (T+1) block cols; each block is l x m x s.
    % -----------------------
    Z0 = tprod(C, B);                      % C ∘ B
    maxIdx = (T + L + 1);                  % need up to Z_{T+L}
    Z = cell(maxIdx+1, 1);                 % store Z_i, i=0..T+L
    Z{1} = Z0;                             % MATLAB 1-based indexing (Z{1} = Z_0)

    Ai = A;                                % A^1
    Z{2} = tprod(tprod(C, Ai), B);         % Z_1

    for i = 2:maxIdx
        Ai = tprod(A, Ai);                 % Ai = A^i
        Z{i+1} = tprod(tprod(C, Ai), B);   % Z_i
    end

    % Assemble H and H_hat
    H  = zeros(nRows, nCols, s);
    H_hat = zeros(nRows, nCols, s);

    % place a block (i,j) with Z_{i+j-2}
    for i = 1:(L+1)
        rowIdx = (i-1)*l + (1:l);
        for j = 1:(T+1)
            colIdx = (j-1)*m + (1:m);
            % H(i,j) = Z_{i+j-2}
            H(rowIdx, colIdx, :)  = Z{i + j - 1};  % since Z{1}=Z_0
            % H1(i,j) = Z_{i+j-1}
            H_hat(rowIdx, colIdx, :) = Z{i + j};      % one-step shift
        end
    end

    % -----------------------
    % Truncated t-SVD of H
    % -----------------------
    [U, S, V] = tsvd(H, 'econ');

    % Truncate to keep (nRows - k) rows and (nCols - k) cols
    U = U(:, 1:r, :);                 % nRows x r_keep_rows x s
    V = V(:, 1:r, :);                 % nCols x r_keep_cols x s
    S = S(1:r, 1:r, :);     % r_keep_rows x r_keep_cols x s

    % -----------------------
    % Form S^{-1/2} slice-wise in Fourier domain 
    % -----------------------
    S_hat = fft(S, [], 3);
    Sinvhalf_hat = zeros(r, r, s);
    for j = 1:s
        Sinvhalf_hat(:,:,j) = S_hat(:,:,j)^(-1/2);
    end
    S_invhalf = ifft(Sinvhalf_hat, [], 3);

    % -----------------------
    % Reduced operators
    %   A_red = S^{-1/2} ∘ U^T ∘ H1 ∘ V ∘ S^{-1/2}
    %   B_red = S^{-1/2} ∘ U^T ∘ H(:, first block column)
    %   C_red = (first block row of H) ∘ V ∘ S^{-1/2}
    % -----------------------
    Ut = tran(U);
    VSinv = tprod(V, S_invhalf);
    SinvUt = tprod(S_invhalf, Ut);

    % A_red
    A_red = tprod(tprod(SinvUt, H_hat), VSinv);            % r x r x s

    % B_red: first block column of H is H(:, 1:m, :)
    H_col1 = H(:, 1:m, :);
    B_red = tprod(SinvUt, H_col1);                      % r x m x s

    % C_red: first block row of H is H(1:l, :, :)
    H_row1 = H(1:l, :, :);
    C_red = tprod(H_row1, VSinv);                       % l x r x s

    
end
