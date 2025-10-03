function [A_red, B_red, C_red] = ERA(A, B, C, k, T, L)
% ERA on the bcirc (block-circulant) system.
%
%   [A_red, B_red, C_red] = ERA(A, B, C, k, T, L)
%
% INPUTS
%   A : n x n x s   state tensor
%   B : n x m x s   
%   C : l x n x s   
%   k : truncation parameter
%   T : (# block columns) minus 1  -> Hankel has (T+1) block columns
%   L : (# block rows)    minus 1  -> Hankel has (L+1) block rows
%


    % -----------------------
    % Validate dimensions & params
    % -----------------------
    [n, n2, s] = size(A);
    if n ~= n2, error('A must be square on modes 1–2.'); end

    if size(B,1) ~= n || size(B,3) ~= s
        error('B must be n x m x s with same n,s as A.');
    end
    if size(C,2) ~= n || size(C,3) ~= s
        error('C must be l x n x s with same n,s as A.');
    end

    m = size(B,2);
    l = size(C,1);

    % Full Hankel sizes (in unfolded/matrix form)
    Nrow_full = l * s * (L+1);      % total rows of H
    Ncol_full = m * s * (T+1);      % total cols of H
    

    r_row = (l*(L+1) - k) * s;      % rows kept after truncation
    r_col = (m*(T+1) - k) * s;      % cols kept after truncation
    
    r = min(r_row,r_col);

    % -----------------------
    % Build bcirc operators ONCE
    % -----------------------
    % NOTE: Dimensions:
    %   As: (n s) x (n s),  Bs: (n s) x (m s),  Cs: (l s) x (n s)
    As = bcirc(A);
    Bs = bcirc(B);
    Cs = bcirc(C);

    % -----------------------
    % Form Z_i = Cs * As^i * Bs, i = 0..(T+L)
    % -----------------------
    Z0 = Cs * Bs;
    maxIdx = T + L + 1;              % need Z_0 ... Z_{T+L}
    Zi = cell(maxIdx+1, 1);
    Zi{1} = Z0;

    Ai = As;                     % As^1
    Zi{2} = Cs * Ai * Bs;        % Z_1
    for i = 2:maxIdx
        Ai = As * Ai;            % Ai = As^i
        Zi{i+1} = Cs * Ai * Bs;  % Z_i
    end

    % -----------------------
    % Assemble block-Hankel matrices H and H1 (shifted)
    % H  (L+1) x (T+1) blocks, each (l s) x (m s)
    % H1 one-step shift: H1(i,j) = G_{i+j-1}
    % -----------------------
    H  = zeros(Nrow_full, Ncol_full);
    H_hat = zeros(Nrow_full, Ncol_full);

    for i = 1:(L+1)
        rows = (i-1)*l*s + (1:l*s);
        for j = 1:(T+1)
            cols = (j-1)*m*s + (1:m*s);
            H(rows, cols)  = Zi{i + j - 1};  % Z_{i+j-2} 
            H_hat(rows, cols) = Zi{i + j};      % Z_{i+j-1}
        end
    end

    % -----------------------
    % Economy SVD of H, then truncate
    % -----------------------
    [U, S, V] = svd(H, 'econ');

    U = U(:, 1:r);                 % (l s (L+1)) x r
    V = V(:, 1:r);                 % (m s (T+1)) x r
    S = S(1:r, 1:r);           % r x r 

    % -----------------------
    % Build S^{-1/2}
    % -----------------------
    S_invhalf = S^(-1/2);   % r x r

    % -----------------------
    % Reduced operators (standard ERA formulas)
    % A_red = S^{-1/2} * U' * H1 * V * S^{-1/2}
    % B_red = S^{-1/2} * U' * (first block column of H)
    % C_red = (first block row of H) * V * S^{-1/2}
    % -----------------------
    Ut = U';                 % r x (l s (L+1))
    VSinv = V * S_invhalf;    % (m s (T+1)) x r
    SinvUt = S_invhalf * Ut;  % r x (l s (L+1))

    % A_red
    A_red = SinvUt * H_hat * VSinv;                % r x r

    % B_red: first block column of H is H(:, 1:(m s))
    B_red = SinvUt * H(:, 1:(m*s));             % r x (m s)

    % C_red: first block row of H is H(1:(l s), :)
    C_red = H(1:(l*s), :) * VSinv;              % (l s) x r
end
