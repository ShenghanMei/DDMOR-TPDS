function [A_red, B_red, C_red] = T_BT(A, B, C, k)
% T_BT  T-product–based balanced truncation (T-BT).
%
%   [A_red, B_red, C_red] = T_BT(A, B, C, k)
%
% Inputs:
%   A : n x n x s   state tensor (mode-3 length = s)
%   B : n x m x s   input tensor
%   C : l x n x s   output tensor
%   k : number of states PER SLICE to remove (so reduced order per slice is n-k)
%
% Outputs (tensor form):
%   A_red : (n-k) x (n-k) x s
%   B_red : (n-k) x  m     x s
%   C_red :  l    x (n-k)  x s
%

    % -----------------------
    % Validate dimensions
    % -----------------------
    [n, n2, s] = size(A);
    if n ~= n2
        error('A must be square on modes 1–2.');
    end
    
    keep = n - k;  % reduced order per slice


    % -----------------------
    % FFT along mode-3 (diagonalizes t-product)
    % -----------------------
    A_hat = fft(A, [], 3);    % n x n x s
    B_hat = fft(B, [], 3);    % n x m x s
    C_hat = fft(C, [], 3);    % l x n x s

    
    Wc_hat = zeros(n, n, s);
    Wo_hat = zeros(n, n, s);
    for j = 1:s
        sys_temp = ss(A_hat(:,:,j),B_hat(:,:,j),C_hat(:,:,j),0,-1);
        Wc_temp = gram(sys_temp,'c');
        Wo_temp = gram(sys_temp,'o'); 
        Wc_hat(:,:,j) = Wc_temp;  % controllability
        Wo_hat(:,:,j) = Wo_temp;  % observability
    end

    Wc = ifft(Wc_hat,[],3);
    Wo = ifft(Wo_hat,[],3);
    [Uc,Sc,Vc]=tsvd(Wc);
    [Uo,So,Vo]=tsvd(Wo);
    
    Sc_hat = fft(Sc,[],3);
    Sc_temp = zeros(size(Sc_hat));

   for i = 1:s
       Sc_temp(:,:,i) = Sc_hat(:,:,i)^(1/2);
   end
   Sc = ifft(Sc_temp,[],3);

   So_hat = fft(So,[],3);
   So_temp = zeros(size(So_hat));

   for i = 1:s
       So_temp(:,:,i) = So_hat(:,:,i)^(1/2);
   end
   So = ifft(So_temp,[],3);

   Zc=tprod(Uc,Sc); % obtain Zc Zo for tensor
   Zo=tprod(Uo,So);
    
    H = tprod(tran(Zo), Zc);          % n x n x s
    [U, S, V] = tsvd(H, 'econ');      % t-SVD

    % Truncate to keep (n-k) tubal singular values
    U = U(:, 1:keep, :);              % n x (n-k) x s
    V = V(:, 1:keep, :);              % n x (n-k) x s
    S = S(1:keep, 1:keep, :);         % (n-k) x (n-k) x s

    % -----------------------
    % Build S^{-1/2} slice-wise in Fourier domain
    % -----------------------
    S_hat = fft(S, [], 3);
    Sinvhalf_hat = zeros(keep, keep, s);
    for j = 1:s
        Sinvhalf_hat(:,:,j) = S_hat(:,:,j)^(-1/2);
    end
    Sinvhalf = ifft(Sinvhalf_hat, [], 3);

    % -----------------------
    % Projection operators (tensor form)
    %   P = Zc ∘ V ∘ S^{-1/2},   Q = Zo ∘ U ∘ S^{-1/2}
    % -----------------------
    P = tprod(tprod(Zc, V), Sinvhalf);   % n x (n-k) x s
    Q = tprod(tprod(Zo, U), Sinvhalf);   % n x (n-k) x s

    % -----------------------
    % Reduced TPDS
    %   A_red = Q^T ∘ A ∘ P
    %   B_red = Q^T ∘ B
    %   C_red = C ∘ P
    % -----------------------
    A_red = tprod(tprod(tran(Q), A), P);     % (n-k) x (n-k) x s
    B_red = tprod(tran(Q), B);               % (n-k) x  m     x s
    C_red = tprod(C, P);                     %  l    x (n-k)  x s
end
