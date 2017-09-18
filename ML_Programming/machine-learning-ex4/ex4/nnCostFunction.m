function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementatien is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%-----------------Part 1: Cost Function --------------------------
% Construct matrix Y as actual output (m, num_labels)
I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1 : m
    Y(i, :) = I(y(i), :);
end
%{
Note: This way gets good result but won't pass review;
      Correct way is to do F.P. within each sample;
A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(size(A2, 1), 1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3); %A3 is hypothetical output; same dimension as Y;
H = A3; %size(5000, 10);

% Use 2 for loop to compute J(theta)
for i = 1 : m
    for k = 1 : num_labels
        J = J + (-Y(i, k) * log(H(i, k)) - (1 - Y(i, k)) * log(1 - H(i, k))); 
    end;
end;
J = J / m;
%}
for i = 1 : m
    % Forward Propogation to calculate h(x)
    a1 = [1 X(i, :)];
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    h = a3;
    y_new = Y(i, :);    
    J = J + (1 / m) * sum(-y_new .* log(h) - (1 - y_new) .* log(1 - h));
    %{
    % Backward Propogation to caculate delta; note delta of last layer is a difference

    delta3 = (h - y_new)';
    delta2 = (Theta2' * delta3)(2 : end) .* sigmoidGradient(z2'); 
    %delta2 = Theta2' * delta3 .* a3 .* (1 - a3);
    big_delta1 = big_delta1 + delta2 * a1;
    big_delta2 = big_delta2 + delta3 * a2;
    Theta1_grad = big_delta1;
    Theta2_grad = big_delta2;
    %}
end;

% Compute Regularized Terms
% note Theta1 25 * 401 -- should ignore col 1, iterate 2->401;
%      Theta2 10 * 26  -- should ignore col 1, iterate 2->26; 
reg = 0;
Theta1_sqr = Theta1 .^ 2;
Theta2_sqr = Theta2 .^ 2;
reg = sum(sum(Theta1_sqr(:, 2:size(Theta1_sqr, 2)))) + sum(sum(Theta2_sqr(:, 2:size(Theta2_sqr, 2)))) 
reg = reg * lambda / (2 * m); 
J = J + reg;

%-----------------Part 2: Gradient --------------------------
big_delta1 = Theta1_grad;
big_delta2 = Theta2_grad;
for i = 1 : m
    %{
    a1 = [1, X(i, :)]';
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    h = a3;

    y_new = zeros(num_labels, 1);
    y_new(y(i)) = 1;

    delta3 = h - y_new;
    delta2 = Theta2' * delta3 .* a2 .* (1 - a2);

    big_delta1 = big_delta1 + delta2(2:end) * a1';
    big_delta2 = big_delta2 + delta3 * a2';
    %}
    a1 = [1 X(i, :)];
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    h = a3;
    
    y_new = zeros(1, num_labels);
    y_new(y(i)) = 1;
   
    d3 = h - y_new;
    % This is a bit different from slides
    % Because d3 a, z are all transposed
    d2 = d3 * Theta2(:, 2 : end) .* sigmoidGradient(z2);
    big_delta1 = big_delta1 + (d2' * a1);
    big_delta2 = big_delta2 + (d3' * a2);
end;
Theta1_grad = big_delta1 / m;
Theta2_grad = big_delta2 / m;


%-----------------Part 3: Gradient with Reg Term--------------------
% This function should add /m for regularized terms which is not cleared explained
% in the slides
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) + lambda / m * Theta1(:, 2 : end);
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) + lambda / m * Theta2(:, 2 : end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
