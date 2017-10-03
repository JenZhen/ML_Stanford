function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for j = 1 : K
    %{ 
    label = ones(m, 1) * j; %[2;2;2;2;2]
    node_j = (idx == label);  %[1;0;0;1;0]
    node_mat = repmat(node_j, 1, n);
    filtered_x = node_mat .* X;
    centroids(j, :) = mean(filtered_x);
    %} 
    row_num = find(idx == j);
    filtered_x = X(row_num, :);
    centroids(j, :) = mean(filtered_x);
    %centroids(j, :) = mean(X([find(idx == j)], :));
end;
    







% =============================================================


end

