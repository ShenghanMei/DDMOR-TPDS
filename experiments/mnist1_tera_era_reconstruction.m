%% MNIST “1” — T-ERA & ERA Reconstruction
clear; clc;

%% ------------------------------------------------------------------------
%  1) Load frames & quick visualization
% -------------------------------------------------------------------------
S = load('Digit1_Frames16.mat');   % file contains variable B: 28x28x16
Frames = S.B;                      % size: 28 x 28 x 16
[h, w, n3] = size(Frames);          

% % 4-D for montage
% Frames4D = reshape(Frames, H, W, 1, K);
% figure; montage(Frames4D, 'Size', [4 4]); colormap gray; axis image off;
% title('Original 16 MNIST “1” Frames','FontSize',16,'FontWeight','bold');

% load a single “impulse” frame for reference
S1 = load('Digit1.mat');   % contains variable R: a single 28x28 image
Impulse = S1.R;
figure; montage(reshape(Impulse, h, w, 1, 1));
colormap gray; axis image off; %title('Reference Impulse','FontSize',14);

%% ------------------------------------------------------------------------
%  2) Reformat frames for T-product shape:  (28*16) x 1 x 28
%     (stack 16 images vertically → 28*16 rows, mode-3 = 28)
% -------------------------------------------------------------------------
TensorSeq = permute(Frames, [1 3 2]);      % 28 x 16 x 28
TensorSeq = reshape(TensorSeq, h*n3, 1, w); % 448 x 1 x 28
TensorSeq = double(TensorSeq); 


%% ------------------------------------------------------------------------
%  3) Hankel tensors H, H_hat from the sequence (tensor case)
%     Here: m=1 input, l=28 outputs, s=28 (mode-3 length)
% -------------------------------------------------------------------------
m = 1;  l = 28;  s = 28;
T = 7; L = 7;

H_TERA = zeros(l*(L+1),m*(T+1),s);
H_hat_TERA = zeros(l*(L+1),m*(T+1),s);


for i = 1:L+1
    for j = 1:T+1
        H_TERA((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = TensorSeq((i+j-2)*l+1:(i+j-1)*l,:,:); 
    end
end

for i = 1:L+1
    for j = 1:T+1
        H_hat_TERA((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = TensorSeq((i+j-1)*l+1:(i+j)*l,:,:); 
    end
end


% Preallocate results:
k_list = 0:7;
numK   = numel(k_list);
Time_ERA   = zeros(numK, 2);    % [T-ERA_time, ERA_time]
Num_ERA    = zeros(numK, 2);    % [T-ERA_paramCount, ERA_paramCount]
%% apply T-ERA to image data, reconstruct image data and calculate relative error
% Hankel block sizes
nRows = l*(L+1);     % total rows in H
nCols = m*(T+1);     % total cols in H

[U_TERA,S_TERA,V_TERA] = tsvd(H_TERA,'econ');

TERA_error = zeros(T+L-1,numK);
image_reconstructed_TERA = zeros(l*(L+T+2),m,s,numK);

for k=0:numK-1
    tStart = tic;
    r_keep_rows = nRows - k;   % rows kept in U,S
    r_keep_cols = nCols - k;   % cols kept in V,S
    
    r = min(r_keep_rows,r_keep_cols);
    
    S_TERA = S_TERA(1:r,1:r,:);
    U_TERA = U_TERA(:,1:r,:);
    V_TERA = V_TERA(:,1:r,:);
    
    % -----------------------
    % Form S^{-1/2} slice-wise in Fourier domain 
    % -----------------------
    S_hat_TERA = fft(S_TERA, [], 3);
    Sinvhalf_hat_TERA = zeros(r, r, s);
    for j = 1:s
        Sinvhalf_hat_TERA(:,:,j) = S_hat_TERA(:,:,j)^(-1/2);
    end
    S_invhalf_TERA = ifft(Sinvhalf_hat_TERA, [], 3);

    A_red_TERA = tprod(tprod(tprod(S_invhalf_TERA,tran(U_TERA)),H_hat_TERA),tprod(V_TERA,S_invhalf_TERA));
    B_red_TERA = tprod(tprod(S_invhalf_TERA,tran(U_TERA)),H_TERA(:,1:m,:));
    C_red_TERA = tprod(H_TERA(1:l,:,:),tprod(V_TERA,S_invhalf_TERA));
    
    Time_ERA(k+1,1) = toc(tStart);
    
    Num_ERA(k+1,1) = numel(A_red_TERA) + numel(B_red_TERA) + numel(C_red_TERA);
    
    image_reconstructed_TERA_temp = zeros(l*(L+T+2),m,s);
    image_reconstructed_TERA_temp(1:l,:,:) = tprod(C_red_TERA,B_red_TERA);
    image_reconstructed_TERA_temp(l+1:2*l,:,:) = tprod(tprod(C_red_TERA,A_red_TERA),B_red_TERA);
    image_reconstructed_TERA_temp = double(image_reconstructed_TERA_temp);
    

    Ak = A_red_TERA;

    for i=1:(T+L)
        Ak = tprod(A_red_TERA,Ak);
        image_reconstructed_TERA_temp((i+1)*l+1:(i+2)*l,:,:) = tprod(tprod(C_red_TERA,Ak),B_red_TERA);
    end
    image_reconstructed_TERA(:,:,:,k+1) = image_reconstructed_TERA_temp;
    
    for i=0:T+L-2
        TERA_error(i+1,k+1)=norm(bcirc(image_reconstructed_TERA_temp(i*l+1:(i+1)*l,:,:))-bcirc(TensorSeq(i*l+1:(i+1)*l,:,:)),'fro')/norm(bcirc(TensorSeq(i*l+1:(i+1)*l,:,:)),'fro');
    end
end

%% reconstruct image
R_TERA = image_reconstructed_TERA(:,:,:,3); % at truncation level k=2
% T: 448 x 1 x 28  (tall stack of 16 images)
[HtimesK, one, w] = size(R_TERA); 
Height = 28;           % image height
n3 = HtimesK / Height;  % number of images = 16 here (checks the 448 = 28*16)

% 1) Undo the last reshape: 448x1x28 -> 28x16x28
tmp = reshape(R_TERA, Height, n3, w);        % size: 28 x 16 x 28

% 2) Undo the earlier permute: 28x16x28 -> 28x28x16
R16 = permute(tmp, [1 3 2]);      % size: 28 x 28 x 16

% visualize the 16 images as a 4x4 montage
R16_4d = reshape(R16, 28, 28, 1, n3);
figure;
montage(R16_4d, 'Size', [4 4]); colormap gray; axis image off;
%title('T-ERA Reconstruction','FontSize', 15);


%%  apply ERA and calculate relative error

H_temp_ERA = zeros(l*s*(L+T+2),m*s);
H_ERA = zeros(l*s*(L+1),m*s*(T+1));
H_hat_ERA = zeros(l*s*(L+1),m*s*(T+1));

for i=1:(T+L+2)
    H_temp_ERA((i-1)*l*s+1:i*l*s,:) = bcirc(TensorSeq((i-1)*l+1:i*l,:,:));
end

for i = 1:L+1
    for j = 1:T+1
        H_ERA((i-1)*l*s+1:i*l*s,(j-1)*m*s+1:j*m*s) = H_temp_ERA((i+j-2)*l*s+1:(i+j-1)*l*s,:); 
    end
end

for i = 1:L+1
    for j = 1:T+1
        H_hat_ERA((i-1)*l*s+1:i*l*s,(j-1)*m*s+1:j*m*s) = H_temp_ERA((i+j-1)*l*s+1:(i+j)*l*s,:); 
    end
end

[U_ERA,S_ERA,V_ERA] = svd(H_ERA,'econ');
ERA_error = zeros(T+L-1,numK);

image_reconstructed_ERA = zeros(l*s*(L+T+2),m*s,numK);
for k=0:numK-1
    tStart = tic;
    
    r_row = (l*(L+1) - k) * s;      % rows kept after truncation
    r_col = (m*(T+1) - k) * s;      % cols kept after truncation
    
    r = min(r_row,r_col);
    
    S_ERA = S_ERA(1:r,1:r);
    U_ERA = U_ERA(:,1:r);
    V_ERA = V_ERA(:,1:r);


    A_red_ERA = S_ERA^(-1/2)*U_ERA'*H_hat_ERA*V_ERA*S_ERA^(-1/2);
    B_red_ERA = S_ERA^(-1/2)*U_ERA'*H_ERA(:,1:m*s);
    C_red_ERA = H_ERA(1:l*s,:)*V_ERA*S_ERA^(-1/2);
    
    Time_ERA(k+1,2) = toc(tStart);
    
    Num_ERA(k+1,2) = numel(A_red_ERA) + numel(B_red_ERA) + numel(C_red_ERA);
    
    image_reconstructed_ERA_temp = zeros(l*s*(L+T+2),m*s);
    image_reconstructed_ERA_temp(1:l*s,:) = C_red_ERA*B_red_ERA;
    image_reconstructed_ERA_temp(l*s+1:2*l*s,:) = C_red_ERA*A_red_ERA*B_red_ERA;

    Ak_ERA = A_red_ERA;

    for i=1:(T+L)
        Ak_ERA = A_red_ERA*Ak_ERA;
        image_reconstructed_ERA_temp((i+1)*l*s+1:(i+2)*l*s,:) = C_red_ERA*Ak_ERA*B_red_ERA;
    end
    
    image_reconstructed_ERA(:,:,k+1) = image_reconstructed_ERA_temp;
    
    for i=0:T+L-2
        ERA_error(i+1,k+1)=norm(image_reconstructed_ERA_temp(i*l*s+1:(i+1)*l*s,:)-H_temp_ERA(i*l*s+1:(i+1)*l*s,:))/norm(H_temp_ERA(i*l*s+1:(i+1)*l*s,:));
    end
end

%% reconstruct image
R_ERA = image_reconstructed_ERA(:,:,3); % at truncation level k=2
R_ERA_temp = R_ERA;
R_ERA = zeros(l*(L+T+2),m,s);
for k=1:(T+L+2)
    R_ERA((k-1)*l+1:k*l,:,:) = block_circulant_matrix_to_tensor(R_ERA_temp((k-1)*l*s+1:k*l*s,:),s);
end

% R_ERA: 448 x 1 x 28  (tall stack of 16 images)
[HtimesK_ERA, one, w_ERA] = size(R_ERA); 
Height_ERA = 28;           % image height
K_ERA = HtimesK_ERA / Height_ERA;  % number of images = 16 here (checks the 448 = 28*16)

% 1) Undo the last reshape: 448x1x28 -> 28x16x28
tmp_ERA = reshape(R_ERA, Height_ERA, K_ERA, w_ERA);        % size: 28 x 16 x 28

% 2) Undo the earlier permute: 28x16x28 -> 28x28x16
R16_ERA = permute(tmp_ERA, [1 3 2]);      % size: 28 x 28 x 16

% visualize the 16 images as a 4x4 montage
R16_4d_ERA = reshape(R16_ERA, 28, 28, 1, K_ERA);
figure;
montage(R16_4d_ERA, 'Size', [4 4]); colormap gray; axis image off;
%title('ERA Reconstruction','FontSize', 15);


%% plot relative error (T-ERA)

% X-axis values
TERA_error_plot = TERA_error(:,2:5); % truncation level k=1,2,3,4
x = 1:7;
colors = lines(size(TERA_error_plot, 2));  % Auto-colors for each line

figure; 

for i = 1:size(TERA_error_plot, 2)
    plot(x, TERA_error_plot(1:7, i), 'o-', ...
         'LineWidth', 2, ...
         'Color', colors(i,:), ...
         'MarkerFaceColor', colors(i,:), ...  % Fills the circle
         'MarkerEdgeColor', colors(i,:)); hold on;     % Optional: match edge to fill
end
axis square
set(gca, 'YScale', 'log');  % Set y-axis to log scale
ax = gca; % Get current axes

% Set font size and weight
ax.FontSize = 13;       % Increase the number size (adjust as needed)
ax.FontWeight = 'bold'; % Make the numbers bold

% Custom x-axis tick labels: 'z0' to 'z6'
xticks(x);  % Set tick positions
xticklabels(arrayfun(@(n) sprintf('Z_{%d}', n), 0:6, 'UniformOutput', false));

% Labels and legend
legend('k=1', 'k=2', 'k=3', 'k=4','Location', 'northwest','FontSize', 14, 'Box', 'off');
xlabel('Snapshots','FontSize', 15);
ylabel('Relative Error','FontSize', 15);
title('T-ERA','FontSize', 15);

%% plot relative error (ERA)

% X-axis values
ERA_error_plot = ERA_error(:,2:5);
x = 1:7;
colors = lines(size(ERA_error, 2));  % Auto-colors for each line

figure; 

for i = 1:size(ERA_error_plot, 2)
    plot(x, ERA_error_plot(1:7, i), 'o-', ...
         'LineWidth', 2, ...
         'Color', colors(i,:), ...
         'MarkerFaceColor', colors(i,:), ...  % Fills the circle
         'MarkerEdgeColor', colors(i,:)); hold on;    % Optional: match edge to fill
end
axis square
set(gca, 'YScale', 'log');  % Set y-axis to log scale
ax = gca; % Get current axes

% Set font size and weight
ax.FontSize = 13;       % Increase the number size (adjust as needed)
ax.FontWeight = 'bold'; % Make the numbers bold

% Custom x-axis tick labels: 'z0' to 'z6'
xticks(x);  % Set tick positions
xticklabels(arrayfun(@(n) sprintf('Z_{%d}', n), 0:6, 'UniformOutput', false));

% Labels and legend
legend('k=1', 'k=2', 'k=3', 'k=4','Location', 'northwest','FontSize', 14, 'Box', 'off');
xlabel('Snapshots','FontSize', 15);
ylabel('Relative Error','FontSize', 15);
title('ERA','FontSize', 15);

%%
%save('MNIST_ERA.mat','Time_ERA', 'Num_ERA')
