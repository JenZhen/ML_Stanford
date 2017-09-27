function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% Following solution doesn't work
% Things I don't understand well:
% size(X_poly) = 12, 9; p = 8 
% I thought there would be a col for X.^0 , ie. col of 1
% Correct way is that first 8 cols are for X.^p respectively;
%{
for col = 2:(p + 1) % col = 1 --> X .^ 0; col = 2 --> X .^1
    X_poly(:, col) = X .^ (col - 1);
end;
%}
for col = 1:p
    X_poly(:, col) = X .^ col;
end;



% =========================================================================

end
