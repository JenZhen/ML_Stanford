function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% X(5000 * 400) Add bias unit (5000 * 401)
%   Theta1(25 * 401) -> X * Theta1' = hiddenlayer(5000 * 25)
% hiddenlayer Add bias unit (5000, 26)
%   Theta2(10 * 26) -> hiddenlayer * Theta2' = output(5000 * 10)
% output: each row is a sample, in which each col is 0 or 1
%   Get the max index position which is the prediction (0 maps to 10)
X = [ones(m, 1) X];
hidden = sigmoid(X * Theta1');
hidden = [ones(size(hidden, 1), 1), hidden];
output = sigmoid(hidden * Theta2');
[max_val, max_idx] = max(output, [], 2);
p = max_idx;








% =========================================================================


end
