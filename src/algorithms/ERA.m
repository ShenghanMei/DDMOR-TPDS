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
    
    num_terms = T + L + 1;                          % need Z_0..Z_{T+L}
    Z = zeros(l*s, m*s, num_terms+1);

    X = Bs;                                     % X_0
    Z(:,:,1) = Cs * X;                     % Z_0
    for t = 1:num_terms
        X = As * X;                             
        Z(:,:,t+1) = Cs * X;               % Z_t
    end
    

    % -----------------------
    % Assemble block-Hankel matrices H and H1 (shifted)
    % H  (L+1) x (T+1) blocks, each (l s) x (m s)
    % H_hat one-step shift: H_hat(i,j) = Z_{i+j-1}
    % -----------------------
    
    idx  = hankel(1:(L+1), (L+1):(L+T+1));      % (L+1) x (T+1), values in 1..(T+L+1)
    idx2 = idx + 1;                              % for shifted Hankel

    % Pull and tile blocks in one shot:
    % Take Z(:,:,idx(:)) which is (l s) x (m s) x ((L+1)*(T+1)).
    ZH   = Z(:,:,idx(:));                   % unshifted blocks
    ZHsh = Z(:,:,idx2(:));                  % shifted blocks

    % Reshape to 4-D: (l s) x (m s) x (L+1) x (T+1)
    ZH   = reshape(ZH,   l*s, m*s, L+1, T+1);
    ZHsh = reshape(ZHsh, l*s, m*s, L+1, T+1);

    % Permute to interleave block rows/cols, then reshape to full H/H1
    H  = reshape(permute(ZH,   [1 3 2 4]), Nrow_full, Ncol_full);    % (l s (L+1)) x (m s (T+1))
    H_hat = reshape(permute(ZHsh, [1 3 2 4]), Nrow_full, Ncol_full);
    
%     H  = zeros(Nrow_full, Ncol_full);
%     H_hat = zeros(Nrow_full, Ncol_full);
% 
%     for i = 1:(L+1)
%         rows = (i-1)*l*s + (1:l*s);
%         for j = 1:(T+1)
%             cols = (j-1)*m*s + (1:m*s);
%             H(rows, cols)  = Zi{i + j - 1};  % Z_{i+j-2} 
%             H_hat(rows, cols) = Zi{i + j};      % Z_{i+j-1}
%         end
%     end

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
    %A_red = S_invhalf * U' * H_hat * V * S_invhalf;                % r x r

    % B_red: first block column of H is H(:, 1:(m s))
    B_red = SinvUt * H(:, 1:(m*s));             % r x (m s)

    % C_red: first block row of H is H(1:(l s), :)
    C_red = H(1:(l*s), :) * VSinv;              % (l s) x r
end
