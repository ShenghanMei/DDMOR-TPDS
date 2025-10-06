function [A_red,B_red,C_red] = BT(A,B,C,k)
% BT  Balanced truncation on the unfolded (bcirc) system.
%
%   [A_red,B_red,C_red] = BT(A,B,C,k)
%
% Inputs
%   A : n x n x r  tensor (mode-3 = tubal length)
%   B : n x m x r  tensor
%   C : l x n x r  tensor
%   k : number of states PER SLICE to remove (so reduced order per slice is n-k)
%
% Outputs (unfolded / matrix form)
%   A_red : ( (n-k)r ) x ( (n-k)r )
%   B_red : ( (n-k)r ) x ( m r )
%   C_red : ( l r )    x ( (n-k)r )


    % -----------------------
    % Dimensions & validation
    % -----------------------
    [n, n2, s] = size(A);
    if n ~= n2
        error('A must be square on modes 1–2.'); 
    end

    % -----------------------
    % Build unfolded (bcirc) system
    % -----------------------
    As = bcirc(A);                % (n r) x (n r)
    Bs = bcirc(B);                % (n r) x (m r)
    Cs = bcirc(C);                % (l r) x (n r)
  
    sys = ss(As, Bs, Cs, 0, -1);

    % Quick stability guard (BT requires stability)
%     if ~isstable(sys)
%         error(['Unstable unfolded system. Stabilize slices of A (e.g., scale to spectral ', ...
%                'radius < 1) or apply stabsep() before BT.']);
%     end

    % -----------------------
    % Gramians 
    % -----------------------
    Wc = gram(sys,'c');
    Wo = gram(sys,'o');

    [Uc,Sc] = svd(Wc, 'econ');  Zc = Uc * Sc^(1/2);
    [Uo,So] = svd(Wo, 'econ');  Zo = Uo * So^(1/2);

    % Hankel 
    H  = Zo' * Zc;                             % (n r) x (n r)
    [U,S,V] = svd(H, 'econ');                   
    
    keep = (n - k) * s;
    U = U(:,1:keep);
    V = V(:,1:keep);
    S = S(1:keep,1:keep);

    % Projection matrices (balanced BT)
    P = Zc * V / sqrt(S);                       % (n r) x ( (n-k) r )
    Q = Zo * U / sqrt(S);                       % (n r) x ( (n-k) r )

    % -----------------------
    % Reduced model (matrix / unfolded form)
    % -----------------------
    A_red = Q.' * As * P;                       % ( (n-k) r ) x ( (n-k) r )
    B_red = Q.' * Bs;                           % ( (n-k) r ) x ( m r )
    C_red = Cs * P;                             % ( l r )     x ( (n-k) r )

end
