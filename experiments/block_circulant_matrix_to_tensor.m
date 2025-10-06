function T = block_circulant_matrix_to_tensor(A, m)
    % block_circulant_matrix_to_tensor - Converts a block circulant matrix into a 3D tensor
    % by cyclically shifting the blocks.
    %
    % Syntax: T = block_circulant_matrix_to_tensor(A, m)
    %
    % Inputs:
    %   A - A block circulant matrix of size k*n x k*n, where k is the number of blocks
    %       and n is the size of each block.
    %   m - The number of slices in the resulting tensor (depth of the tensor).
    %
    % Output:
    %   T - The resulting 3D tensor (block circulant matrix transformed to tensor).

    % Get the size of the block circulant matrix A
    [sizeA1, sizeA2] = size(A);
    
    % Ensure A is a square matrix
%     if sizeA1 ~= sizeA2
%         error('Input matrix A must be square');
%     end
    
    % Determine the number of blocks (k) and block size (n)
    % The matrix A should be of size k*n x k*n, where n is the block size
    n1 = sizeA1 / m;  % Size of each block (assuming square blocks)
    n2 = sizeA2 / m;  % Number of blocks
    
%     % Check if the matrix dimensions are compatible for a block circulant matrix
%     if mod(sizeA1, n) ~= 0
%         error('Matrix size is not divisible by the block size');
%     end
    
    % Initialize the 3D tensor with zeros
    T = zeros(n1, n2, m);
    
    % Populate the tensor with cyclically shifted versions of the block circulant matrix
    for i = 1:m
        % Shift the matrix for each slice
        T(:,:,i) = A(n1*(i-1)+1:n1*i,1:n2);  % Shift rows and columns
    end
end

