function [A_red, B_red, C_red] = T_POD(A, B, C, k, T, L)
% T_POD model reduction.
%
%   [A_red, B_red, C_red] = T_POD(A, B, C, k, T, L)
%
% Inputs
%   A : n x n x s   state tensor (mode-3 length = s)
%   B : n x m x s   control tensor
%   C : l x n x s   output tensor
%   k : truncation parameter

%
% Outputs (tensor form)
%   A_red : r x r x s
%   B_red : r x m x s
%   C_red : l x r x s
%
% Method 
%   Build block “snapshot” tensors:
%     X = [B, A∘B, A^2∘B, ..., A^T∘B]          ∈ R^{n × m(T+1) × s}
%     Y = [C^T, (A^T)∘C^T, ..., (A^T)^L∘C^T]   ∈ R^{n × l(L+1) × s}
%   Form H = Y^T ∘ X  ∈ R^{l(L+1) × m(T+1) × s}, take t-SVD H ≈ U ∘ S ∘ V^T,
%   truncate to r, build S^{-1/2} slice-wise, then
%     P = X ∘ V ∘ S^{-1/2},   Q = Y ∘ U ∘ S^{-1/2}
%     A_red = Q^T ∘ A ∘ P,   B_red = Q^T ∘ B,   C_red = C ∘ P.

    % -------- Validate dimensions --------
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

    % Target retained dimensions on each side; pick r as the common kept rank
    keep_Y = l*(L+1) - k;
    keep_X = m*(T+1) - k;
 
    r = min(keep_Y, keep_X);

    % -------- Build X = [B, A∘B, ..., A^T∘B] --------
    X = zeros(n, m*(T+1), s);
    temp = B;                                % A^0 ∘ B
    X(:, 1:m, :) = temp;
    for t = 2:(T+1)
        temp = tprod(A, temp);               % temp = A ∘ temp
        cols = (t-1)*m + (1:m);
        X(:, cols, :) = temp;
    end

    % -------- Build Y = [C^T, (A^T)∘C^T, ..., (A^T)^L∘C^T] --------
    XposeA = tran(A);                        % A^T in t-algebra
    Cpose  = tran(C);                        % C^T  (n x l x s)
    Y = zeros(n, l*(L+1), s);
    temp = Cpose;                            % (A^T)^0 ∘ C^T
    Y(:, 1:l, :) = temp;
    for t = 2:(L+1)
        temp = tprod(XposeA, temp);          % temp = A^T ∘ temp
        cols = (t-1)*l + (1:l);
        Y(:, cols, :) = temp;
    end

    % -------- Cross-correlation H = Y^T ∘ X and its t-SVD --------
    H = tprod(tran(Y), X);                   % size: l(L+1) x m(T+1) x s
    [U, S, V] = tsvd(H, 'econ');

    % Truncate to rank r in t-SVD sense
    U = U(:, 1:r, :);                        % l(L+1) x r x s
    V = V(:, 1:r, :);                        % m(T+1) x r x s
    S = S(1:r, 1:r, :);                      % r x r x s

    % -------- Build S^{-1/2} (slice-wise, robust) --------
    % Work in the Fourier domain; each slice should be Hermitian PSD.
    S_hat = fft(S, [], 3);
    Sinvhalf_hat = zeros(r, r, s);
    for j = 1:s
        Sinvhalf_hat(:,:,j) = S_hat(:,:,j)^(-1/2);
    end
    S_invhalf = ifft(Sinvhalf_hat, [], 3);

    % -------- Projections: P = X ∘ V ∘ S^{-1/2},  Q = Y ∘ U ∘ S^{-1/2} --------
    P = tprod(tprod(X, V), S_invhalf);       % n x r x s
    Q = tprod(tprod(Y, U), S_invhalf);       % n x r x s

    % -------- Reduced operators --------
    %   A_red = Q^T ∘ A ∘ P
    %   B_red = Q^T ∘ B
    %   C_red = C ∘ P
    A_red = tprod(tprod(tran(Q), A), P);     % r x r x s
    B_red = tprod(tran(Q), B);               % r x m x s
    C_red = tprod(C, P);                     % l x r x s
end
