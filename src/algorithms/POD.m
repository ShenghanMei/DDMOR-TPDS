function [A_red, B_red, C_red] = POD(A, B, C, k, T, L)
% POD  Proper Orthogonal Decomposition (bcirc formulation)
%
%   [A_red, B_red, C_red] = POD(A, B, C, k, T, L)
%
% Inputs
%   A : n x n x s   state tensor        (mode-3 length = s)
%   B : n x m x s   
%   C : l x n x s   
%   k : truncation parameter (keep r = min(l*(L+1)-k, m*(T+1)-k) * s )

%
% Outputs (matrix / unfolded form)
%   A_red : r x r
%   B_red : r x (m*s)
%   C_red : (l*s) x r
%
% Method (unfolded POD):
%   Build unfolded snapshots
%     X = [B, A B, A^2 B, ..., A^T B]              ∈ R^{(n s) × (m s (T+1))}
%     Y = [C^T, A^T C^T, ..., (A^T)^L C^T]         ∈ R^{(n s) × (l s (L+1))}
%   Form H = Y^T X ∈ R^{(l s (L+1)) × (m s (T+1))}, SVD(H)=U S V'.
%   With r = min(l(L+1)-k, m(T+1)-k)*s:
%     P = X V S^{-1/2},  Q = Y U S^{-1/2},
%     A_red = Q' A P,  B_red = Q' B,  C_red = C P.
%

    % ---------- Validate tensor shapes ----------
    [n, n2, s] = size(A);
    if n ~= n2, error('A must be square on modes 1–2.'); end
    if size(B,1) ~= n || size(B,3) ~= s
        error('B must be n x m x s with the same n,s as A.');
    end
    if size(C,2) ~= n || size(C,3) ~= s
        error('C must be l x n x s with the same n,s as A.');
    end

    m = size(B,2);
    l = size(C,1);

    % ---------- Unfold (block-circulant) operators ----------
    % Sizes: bA:(n s)x(n s), bB:(n s)x(m s), bC:(l s)x(n s)
    As = bcirc(A);
    Bs = bcirc(B);
    Cs = bcirc(C);

    Ns   = n * s;                 % unfolded state dimension
    Xcol = m * s * (T+1);         % columns in X
    Ycol = l * s * (L+1);         % columns in Y

    % Target kept ranks from each side, then overall r
    keep_Y = l*(L+1) - k;
    keep_X = m*(T+1) - k;
   
    r = min(keep_Y, keep_X) * s;  % overall kept in unfolded space

    % ---------- Build snapshot matrices X and Y with preallocation ----------
    % X = [B, A B, A^2 B, ..., A^T B]   (Ns x MsTs)
    X = zeros(Ns, Xcol);
    tmp = Bs;
    X(:, 1:(m*s)) = tmp;
    for t = 2:(T+1)
        tmp = As * tmp;                                % tmp = A^t B
        cols = (t-1)*m*s + (1:m*s);
        X(:, cols) = tmp;
    end

    % Y = [C^T, A^T C^T, ..., (A^T)^L C^T]   (Ns x LsLs)
    Y = zeros(Ns, Ycol);
    tmp = Cs';                                       % C^T
    Y(:, 1:(l*s)) = tmp;
    AsT = As';                                       % A^T
    for t = 2:(L+1)
        tmp = AsT * tmp;                              % tmp = (A^T)^t C^T
        cols = (t-1)*l*s + (1:l*s);
        Y(:, cols) = tmp;
    end

    % ---------- Cross Gramian-like matrix and SVD ----------
    H = Y' * X;        % (l s (L+1)) x (m s (T+1))
    %H_hat = Y' * As * X;
    [U, S, V] = svd(H, 'econ');
   

    % ---------- Truncate to r ----------
   
    U = U(:, 1:r);
    V = V(:, 1:r);
    S = S(1:r, 1:r);

    % ---------- Build S^{-1/2} ----------
    S_invhalf = S^(-1/2);           % r x r

    % ---------- Projections ----------
    % P = X * V * S^{-1/2},   Q = Y * U * S^{-1/2}
    P = X * V * S_invhalf;                       % (Ns x r)
    Q = Y * U * S_invhalf;                       % (Ns x r)

    % ---------- Reduced operators ----------
    % A_red = Q' * A * P,   B_red = Q' * B,   C_red = C * P
    A_red = Q' * As * P;  % r x r
    %A_red = S_invhalf * U' * H_hat * V * S_invhalf;
    B_red = Q' * Bs;                                 % r x (m s)
    C_red = Cs * P;                                   % (l s) x r
end
