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


K = num_labels;
X = [ones(m, 1), X];
% Calculate activations and h_Theta(x)
% Remember to add bias units to a1 and a2
a1 = X;
z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)];
z3 = a2 * Theta2';
hx = sigmoid(z3);

% loop over all labels (k)
for k = 1:K
    % 1) Get yk where k == some label (1,2,...10)
    % 2) Get h_Theta(x)_k
    % 3) Calculate J_k and add to J
    yk = y == k;
    hx_k = hx(:, k);
    J = J + sum(-yk.*log(hx_k) - (1-yk).*log(1-hx_k))/m;
end

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Backpropagation
for t = 1:m
    % Forward pass => activations
    a1b = X(t, :);
    z2b = a1b * Theta1';
    a2b = [1, sigmoid(z2b)];
    z3b = a2b * Theta2';
    hxb = sigmoid(z3b);
    
    % For each node j in layer l compute delta
    yt = (1:num_labels) == y(t);
    delta3 = hxb - yt;
    delta2 = delta3 * Theta2 .* [1, sigmoidGradient(z2b)];
    delta2 = delta2(2:end); % skip delta_0^(2)
    Theta1_grad = Theta1_grad + delta2' * a1b;
    Theta2_grad = Theta2_grad + delta3' * a2b;
end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Regularization of cost function is sum of all matrix elements squared,
% except first column
reg = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2));
J = J + lambda*reg/(2*m);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda * Theta2(:, 2:end);

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
