clear
clc
n = 100; m = 2;
l = 2; r = 9;
A = 1/200*randn(n,n,r);  % Vector for circulant matrix
B = randn(n,m,r);
C = randn(l,n,r);

sys = ss(bcirc(A),bcirc(B),bcirc(C),0,-1);

rng('default');
t = zeros(10,1); t1 = zeros(10,1);
mm = zeros(10,1);
tm = zeros(10,1);mr = zeros(10,1);tr = zeros(10,1);
a = 50;
%a is T+L
v = [55,60,65,70,75,80,85,90,95,100];
for z = 1:10
q = v(z);
tic;
temp = B;
X = [];
for i = 1:a
X = [X temp];
temp = tprod(A,temp);
end

A1 = tran(A); C1 = tran(C);
Y = [];
temp = C1;
for i = 1:a
Y = [Y temp];
temp = tprod(A1,temp);
end


H = tprod(tran(Y),X);
[U,S,V] = tsvd(H);
S = S(1:100-q,1:100-q,:);
U = U(:,1:100-q,:);
V = V(:,1:100-q,:);
S1 = fft(S,[],3);
S_hat = zeros(100-q,100-q,2);
for i = 1:r
    S_hat(:,:,i) = S1(:,:,i)^(-1/2);
    S2 = ifft(S_hat,[],3);
end

P1 = tprod(X,V);
Q1 = tprod(Y,U);
P = tprod(P1,S2);
Q = tprod(Q1,S2);
clearvars -except n m l r A B C sys t t1 mm tm q mr tr a v z P Q 
vars_before = whos;

Ar = tprod(tprod(tran(Q),A),P);
Br = tprod(tran(Q),B);
Cr = tprod(C,P);
vars_after = whos;
t1(z,1) = toc
sys2 = ss(bcirc(Ar),bcirc(Br),bcirc(Cr),0,-1);

tm(z,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0);
tr(z,1) = hinfnorm(sys-sys2)/hinfnorm(sys); % Calculate relative error for tensor


%clearvars -except n m l r A B C sys t t1 mm tm q mr tr a v z
%matrix
%vars_before = whos;
tic;
bB = bcirc(B); bA = bcirc(A); bC = bcirc(C);
bX = [];
temp = bB;
for i = 1:a
    bX = [bX temp];
    temp = bA* temp;
end

bY = [];
bA1 = tran(bA); bC1 = tran(bC);
temp = bC1;
for i = 1:a
    bY = [bY temp];
    temp = bA1 * temp;
end

bH = tran(bY)*bX;
[bU,bS,bV] = svd(bH);
bS = bS(1:(100-q)*r,1:(100-q)*r);
bU = bU(:,1:(100-q)*r);
bV = bV(:,1:(100-q)*r);
bP = bX*bV*bS^(-1/2);
bQ = bY*bU*bS^(-1/2);

clearvars -except n m l r A B C sys t t1 mm tm q mr tr a v z bQ bA bP bC bB
vars_before = whos;
bAr = tran(bQ)*bA*bP;
bBr = tran(bQ)*bB;
bCr = bC*bP;
vars_after = whos;
t(z,1) = toc;
mm(z,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0); % Prevent negative values
sys_m = ss(bAr,bBr,bCr,0,-1);
mr(z,1) = hinfnorm(sys-sys_m)/hinfnorm(sys);
fprintf("Matrix Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e \n", t(z,1), mm(z,1) /1e6,mr(z,1));
fprintf("Tensor Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e \n", t1(z,1), tm(z,1) /1e6,tr(z,1));
end