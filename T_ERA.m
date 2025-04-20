clear
clc
n = 100; m = 5;
l = 5; r = 9;
T = 19; L = 19;
A = 1/200*randn(n,n,r);  
B = randn(n,m,r);
C = randn(l,n,r);
sys = ss(bcirc(A),bcirc(B),bcirc(C),0,-1);
t = zeros(n/10,1);
t1 = zeros(n/10,1);
mm = zeros(n/10,1);
tm = zeros(n/10,1);mr = zeros(n/10,1);tr = zeros(n/10,1);
for z = 1:n/10
    q = n/2 + z*5;

tic;
H_temp = zeros(l*(L+T+2),m,r);
H = zeros(l*(L+1),m*(T+1),r);
H1 = zeros(l*(L+1),m*(T+1),r);
H_temp(1:l,:,:) = tprod(C,B);
H_temp(l+1:2*l,:,:) = tprod(tprod(C,A),B);

Ak = A;

for k=1:(T+L)
    Ak = tprod(A,Ak);
    H_temp((k+1)*l+1:(k+2)*l,:,:) = tprod(tprod(C,Ak),B);
end

for i = 1:L+1
    for j = 1:T+1
        H((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = H_temp((i+j-2)*l+1:(i+j-1)*l,:,:); 
    end
end

for i = 1:L+1
    for j = 1:T+1
        H1((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = H_temp((i+j-1)*l+1:(i+j)*l,:,:); 
    end
end

[U,S,V] = tsvd(H,'econ');
S = S(1:(l*(L+1)-q),1:(m*(T+1)-q),:);
U = U(:,1:l*(L+1)-q,:);
V = V(:,1:m*(T+1)-q,:);

S1 = fft(S,[],3);
S_hat = zeros(l*(L+1)-q,m*(T+1)-q,r);
for i = 1:r
    S_hat(:,:,i) = S1(:,:,i)^(-1/2);
    S2 = ifft(S_hat,[],3);
end
clearvars -except A B C n m l r T L sys t t1 mm tm mr tr z q S2 U H1 V H
vars_before = whos;
Ar = tprod(tprod(tprod(S2,tran(U)),H1),tprod(V,S2));
Br = tprod(tprod(S2,tran(U)),H(:,1:m,:));
Cr = tprod(H(1:l,:,:),tprod(V,S2));
vars_after = whos;
t1(z,1) = toc
sys2 = ss(bcirc(Ar),bcirc(Br),bcirc(Cr),0,-1);
tm(z,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0);
tr(z,1) = hinfnorm(sys-sys2)/hinfnorm(sys);
%  matrix

tic
A1 = bcirc(A);
B1 = bcirc(B);
C1 = bcirc(C);

H_temp_m = zeros(l*r*(L+T+2),m*r);
H_m = zeros(l*r*(L+1),m*r*(T+1));
H1_m = zeros(l*r*(L+1),m*r*(T+1));
H_temp_m(1:l*r,:) = C1*B1;
H_temp_m(l*r+1:2*l*r,:) = C1*A1*B1;

Ak_m = A1;

for k=1:(T+L)
    Ak_m = A1*Ak_m;
    H_temp_m((k+1)*l*r+1:(k+2)*l*r,:) = C1*Ak_m*B1;
end

for i = 1:L+1
    for j = 1:T+1
        H_m((i-1)*l*r+1:i*l*r,(j-1)*m*r+1:j*m*r) = H_temp_m((i+j-2)*l*r+1:(i+j-1)*l*r,:); 
    end
end

for i = 1:L+1
    for j = 1:T+1
        H1_m((i-1)*l*r+1:i*l*r,(j-1)*m*r+1:j*m*r) = H_temp_m((i+j-1)*l*r+1:(i+j)*l*r,:); 
    end
end

[U_m,S_m,V_m] = svd(H_m,'econ');
S_m = S_m(1:((l*(L+1)-q)*r),1:((m*(T+1)-q)*r));
U_m = U_m(:,1:(l*(L+1)-q)*r);
V_m = V_m(:,1:(m*(T+1)-q)*r);

clearvars -except A B C n m l r T L sys t t1 mm tm mr tr z q S_m U_m H1_m V_m H_m
vars_before = whos;
Ar_m = S_m^(-1/2)*U_m'*H1_m*V_m*S_m^(-1/2);
Br_m = S_m^(-1/2)*U_m'*H_m(:,1:m*r);
Cr_m = H_m(1:l*r,:)*V_m*S_m^(-1/2);
vars_after = whos;
t(z,1) = toc
sys_m = ss(Ar_m,Br_m,Cr_m,0,-1);
mm(z,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0);

mr(z,1) = hinfnorm(sys-sys_m)/hinfnorm(sys);
fprintf("Matrix Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e\n", t(z,1), mm(z,1) /1e6,mr(z,1));
fprintf("Tensor Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e\n", t1(z,1), tm(z,1) /1e6,tr(z,1));
end