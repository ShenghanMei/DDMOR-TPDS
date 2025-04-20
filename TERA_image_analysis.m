clear
clc
%% parameter
m = 5;
l = 5; r = 3;
T = 10; L = 10;

%% use video data to construct Hankel tensor 
load('Image_data.mat');

H = zeros(l*(L+1),m*(T+1),r);
H1 = zeros(l*(L+1),m*(T+1),r);


for i = 1:L+1
    for j = 1:T+1
        H((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = image_data((i+j-2)*l+1:(i+j-1)*l,:,:); 
    end
end

for i = 1:L+1
    for j = 1:T+1
        H1((i-1)*l+1:i*l,(j-1)*m+1:j*m,:) = image_data((i+j-1)*l+1:(i+j)*l,:,:); 
    end
end


%% apply T-ERA to image data, reconstruct image data and calculate relative error

[U,S,V] = tsvd(H,'econ');
tensor_error = zeros(T+L+1,6);
for q=0:10:50
    
    S = S(1:(l*(L+1)-q),1:(m*(T+1)-q),:);
    U = U(:,1:l*(L+1)-q,:);
    V = V(:,1:m*(T+1)-q,:);

    S1 = fft(S,[],3);
    S_hat = zeros(l*(L+1)-q,m*(T+1)-q,r);
    for i = 1:r
        S_hat(:,:,i) = S1(:,:,i)^(-1/2);
        S2 = ifft(S_hat,[],3);
    end
    Ar = tprod(tprod(tprod(S2,tran(U)),H1),tprod(V,S2));
    Br = tprod(tprod(S2,tran(U)),H(:,1:m,:));
    Cr = tprod(H(1:l,:,:),tprod(V,S2));
    
    image_reconstructed = zeros(l*(L+T+2),m,r);
    image_reconstructed(1:l,:,:) = tprod(Cr,Br);
    image_reconstructed(l+1:2*l,:,:) = tprod(tprod(Cr,Ar),Br);

    Ak = Ar;

    for k=1:(T+L)
        Ak = tprod(Ar,Ak);
        image_reconstructed((k+1)*l+1:(k+2)*l,:,:) = tprod(tprod(Cr,Ak),Br);
    end
    
    for i=0:T+L
        tensor_error(i+1,q/10+1)=norm(bcirc(image_reconstructed(i*l+1:(i+1)*l,:,:))-bcirc(image_data(i*l+1:(i+1)*l,:,:)))/norm(bcirc(image_data(i*l+1:(i+1)*l,:,:)));
    end
end

%%  apply ERA and calculate relative error

H_temp_m = zeros(l*r*(L+T+2),m*r);
H_m = zeros(l*r*(L+1),m*r*(T+1));
H1_m = zeros(l*r*(L+1),m*r*(T+1));

for k=1:(T+L+2)
    H_temp_m((k-1)*l*r+1:k*l*r,:) = bcirc(image_data((k-1)*l+1:k*l,:,:));
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
matrix_error = zeros(T+L+1,6);
for q=0:10:50
    S_m = S_m(1:((l*(L+1)-q)*r),1:((m*(T+1)-q)*r));
    U_m = U_m(:,1:(l*(L+1)-q)*r);
    V_m = V_m(:,1:(m*(T+1)-q)*r);


    Ar_m = S_m^(-1/2)*U_m'*H1_m*V_m*S_m^(-1/2);
    Br_m = S_m^(-1/2)*U_m'*H_m(:,1:m*r);
    Cr_m = H_m(1:l*r,:)*V_m*S_m^(-1/2);
    
    image_reconstructed_m = zeros(l*r*(L+T+2),m*r);
    image_reconstructed_m(1:l*r,:) = Cr_m*Br_m;
    image_reconstructed_m(l*r+1:2*l*r,:) = Cr_m*Ar_m*Br_m;

    Ak_m = Ar_m;

    for k=1:(T+L)
        Ak_m = Ar_m*Ak_m;
        image_reconstructed_m((k+1)*l*r+1:(k+2)*l*r,:) = Cr_m*Ak_m*Br_m;
    end
    
    for i=0:T+L
        matrix_error(i+1,q/10+1)=norm(image_reconstructed_m(i*l*r+1:(i+1)*l*r,:)-H_temp_m(i*l*r+1:(i+1)*l*r,:))/norm(H_temp_m(i*l*r+1:(i+1)*l*r,:));
    end
end


%% plot relative error (tensor)

% X-axis values
tensor_error = tensor_error(:,3:6);
x = 1:10;
colors = lines(size(tensor_error, 2));  % Auto-colors for each line

figure; 

for i = 1:size(tensor_error, 2)
    plot(x, tensor_error(1:10, i), 'o-', ...
         'LineWidth', 2, ...
         'Color', colors(i,:), ...
         'MarkerFaceColor', colors(i,:), ...  % Fills the circle
         'MarkerEdgeColor', colors(i,:)); hold on;     % Optional: match edge to fill
end
set(gca, 'YScale', 'log');  % Set y-axis to log scale
ax = gca; % Get current axes

% Set font size and weight
ax.FontSize = 11;       % Increase the number size (adjust as needed)
ax.FontWeight = 'bold'; % Make the numbers bold

% Custom x-axis tick labels: 'z0' to 'z9'
xticks(x);  % Set tick positions
xticklabels(arrayfun(@(n) sprintf('Z_{%d}', n), 0:9, 'UniformOutput', false));

% Labels and legend
legend('q=20', 'q=30', 'q=40', 'q=50','Location', 'northwest','FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');
xlabel('Snapshots','FontSize', 15, 'FontWeight', 'bold');
ylabel('Relative Error','FontSize', 15, 'FontWeight', 'bold');
title('T-ERA');

%% plot relative error (matrix)

% X-axis values
matrix_error = matrix_error(:,3:6);
x = 1:10;
colors = lines(size(matrix_error, 2));  % Auto-colors for each line

figure;

for i = 1:size(matrix_error, 2)
    plot(x, matrix_error(1:10, i), 'o-', ...
         'LineWidth', 2, ...
         'Color', colors(i,:), ...
         'MarkerFaceColor', colors(i,:), ...  % Fills the circle
         'MarkerEdgeColor', colors(i,:)); hold on;    % Optional: match edge to fill
end
set(gca, 'YScale', 'log');  % Set y-axis to log scale
ax = gca; % Get current axes

% Set font size and weight
ax.FontSize = 11;       % Increase the number size (adjust as needed)
ax.FontWeight = 'bold'; % Make the numbers bold

% Custom x-axis tick labels: 'z0' to 'z9'
xticks(x);  % Set tick positions
xticklabels(arrayfun(@(n) sprintf('Z_{%d}', n), 0:9, 'UniformOutput', false));

% Labels and legend
legend('q=20', 'q=30', 'q=40', 'q=50','Location', 'northwest','FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');
xlabel('Snapshots','FontSize', 15, 'FontWeight', 'bold');
ylabel('Relative Error','FontSize', 15, 'FontWeight', 'bold');
title('ERA');
