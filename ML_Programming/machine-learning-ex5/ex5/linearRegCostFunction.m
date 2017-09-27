function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%{
h = X * theta;
err = h - y;
J_nonReg = sum(err .^ 2) / (2 * m);
theta_sqr = theta(2: size(theta)) .^ 2;
J_Reg = J_nonReg + lambda * sum(theta_sqr) / (2 * m);
J = J_Reg;

grad_nonReg = (X' * (h - y)) / m;
regTerm = lambda * theta(2: size(theta)) / m;
grad_Reg = grad_nonReg + [0; regTerm];
grad = grad_Reg;
%}

h = X * theta;
J = sum((h - y) .^ 2) / (2*m) + lambda * sum(theta(2:end) .^ 2) / (2*m);
grad = X' * (h - y) / m + lambda * [0; theta(2:end)] / m; 









% =========================================================================

grad = grad(:);

end
