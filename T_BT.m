clear
clc

n = 100; 
l = 4; 
m = 5; r = 9;
rng(2);

A = 1/200*randn(n,n,r);  
B = randn(n,m,r);
C = randn(l,n,r);
sys = ss(bcirc(A),bcirc(B),bcirc(C),0,-1);


Wc = gram(sys,'c');
Wo = gram(sys,'o');
t = zeros(n/10,1);
t1 = zeros(n/10,1);
mm = zeros(n/10,1);
tm = zeros(n/10,1);mr = zeros(n/10,1);tr = zeros(n/10,1);

v = [55,60,65,70,75,80,85,90,95,100];
qt = n - v/r;
%%%%%% matrix 
for j = 1:10
   q = v(j);
   %clearvars -except n m l r A B C sys Wc Wo t t1 mm tm j qt v q mr tr
   % Measure memory before matrix computation
   tic
   [Uc,Sc,Vc]=svd(Wc);
   [Uo,So,Vo]=svd(Wo);

   Zc=Uc*Sc^(1/2);
   Zo=Uo*So^(1/2);

   Zc_m = Zc; % obtain Zc Zo for matrix
   Zo_m = Zo;

   H_m = Zo_m'*Zc_m;
   [U_m,S_m,V_m] = svd(H_m,"econ");

   S_m = S_m(1:(n-q)*r,1:(n-q)*r);
   U_m = U_m(1:n*r,1:(n-q)*r);
   V_m = V_m(1:n*r,1:(n-q)*r);

   P_m = Zc_m*V_m*(S_m^(-1/2));
   Q_m = Zo_m*U_m*(S_m^(-1/2));
   clearvars -except n m l r A B C sys Wc Wo t t1 mm tm j qt v q mr tr Q_m P_m
   vars_before = whos; % Track all variables
   Ar_m = Q_m'*bcirc(A)*P_m;
   Br_m = Q_m'*bcirc(B);
   Cr_m = bcirc(C)*P_m;
   vars_after = whos;
   t(j,1) = toc

   % Measure memory after matrix computation
   
   mm(j,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0); % Prevent negative values
   sys_m = ss(Ar_m,Br_m,Cr_m,0,-1);
   mr(j,1) = hinfnorm(sys-sys_m)/hinfnorm(sys);
   % Measure memory before tensor computation
   %clearvars -except n m l r A B C sys Wc Wo t t1 mm tm j qt v q mr tr
   %vars_before = whos;
   tic;

   Wc_t = block_circulant_matrix_to_tensor(Wc, r);
   Wo_t = block_circulant_matrix_to_tensor(Wo, r);

   [Uc_t,Sc_t,Vc_t]=tsvd(Wc_t);
   [Uo_t,So_t,Vo_t]=tsvd(Wo_t);
   Sc1_t = fft(Sc_t,[],3);

   for i = 1:r
       Sc_hat(:,:,i) = Sc1_t(:,:,i)^(1/2);
       Sc2_t = ifft(Sc_hat,[],3);
   end

   So1_t = fft(So_t,[],3);

   for i = 1:r
       So_hat(:,:,i) = So1_t(:,:,i)^(1/2);
       So2_t = ifft(So_hat,[],3);
   end

   Zc_t=tprod(Uc_t,Sc2_t); % obtain Zc Zo for tensor
   Zo_t=tprod(Uo_t,So2_t);

   H = tprod(tran(Zo_t), Zc_t);
   [U,S,V] = tsvd(H,"econ");

   for i =1:n
       w = S(i,i,:);
       norm(w(:));
   end

   S = S(1:n-q,1:n-q,:);
   U = U(1:n,1:n-q,:);
   V = V(1:n,1:n-q,:);
   S1 = fft(S,[],3);
   S_hat = zeros(n-q,n-q,r);

   for i = 1:r
       S_hat(:,:,i) = S1(:,:,i)^(-1/2);
       S2 = ifft(S_hat,[],3);
   end

   P1 = tprod(Zc_t,V);
   Q1 = tprod(Zo_t,U);
   P = tprod(P1,S2);
   Q = tprod(Q1,S2);
   clearvars -except n m l r A B C sys Wc Wo t t1 mm tm j qt v q mr tr Q P
   vars_before = whos;
   Ar = tprod(tprod(tran(Q),A),P);
   Br = tprod(tran(Q),B);
   Cr = tprod(C,P);
   vars_after = whos;
   t1(j,1) = toc
   tm(j,1) = max(sum([vars_after.bytes]) - sum([vars_before.bytes]), 0); % Prevent negative values
   sys2 = ss(bcirc(Ar),bcirc(Br),bcirc(Cr),0,-1);
   % Measure memory after tensor computation
   tr(j,1) = hinfnorm(sys-sys2)/hinfnorm(sys);
   fprintf("Matrix Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e \n", t(j,1), mm(j,1) /1e6,mr(j,1));
   fprintf("Tensor Method: Time = %.4f sec, Memory = %.2f MB, relative error = %.4e \n", t1(j,1), tm(j,1) /1e6,tr(j,1));
end