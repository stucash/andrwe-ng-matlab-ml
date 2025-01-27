function [J, grad] = nnCostFunction(nn_params, ...
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
K = num_labels;

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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% prep work - forward prop (Vectorised)
a1w1 = [ones(m,1) X];  % 5000 x 401
Z2 = Theta1 * a1w1'; % 25 x 5000
a2 = sigmoid(Z2);  % hidden layer activation units, input to next layer
a2w1 = [ones(length(a2),1)'; a2];  % 26 x 5000
Z3 = Theta2 * a2w1; % 10 x 5000
a3 = sigmoid(Z3); % output: 10 x 5000
Ky = y == (1:K); % relabel y to contain only binary value for each class.

% Part 1.1 - cost term, y:5000x1, ky:5000x10, a3:10 x 5000, K=10
Cost_one = (-Ky') .* log(a3);  % 10 x 5000
Cost_zero = (1-Ky') .* log(1 - a3);  % 10 x 5000
Cost = sum(sum(Cost_one - Cost_zero, 2));
J = (1/m) * Cost;

% Alternative vectorised solution
% This solution does some unnecessary calculations therefore not preferred.
% Cost_one = (-Ky) * log(a3);
% Cost_zero = (1-Ky) * log(1 - a3);
% Cost = diag(Cost_one) - diag(Cost_zero); % cost of each class.
% J = (1/m) * sum(Cost);

% Part 1.2 - regularised term
Theta1_rest = Theta1(:, 2: (input_layer_size+1));
Theta2_rest = Theta2(:, 2: (hidden_layer_size+1));
Theta = [Theta1_rest(:); Theta2_rest(:)];  % all theta in one column vector
R = (lambda / (2*m)) * (Theta' * Theta);

J = J + R;

% Part 2 Back prop (Vectorised)
E3 = a3 - Ky';  % 10 x 5000
E2 = (E3' * Theta2_rest) .* sigmoidGradient(Z2');  % 5000 x 25

% this function assumes bias units' errors part of calculation.
Theta2_grad = (1/m) * (E3 * a2w1');  % 10 x 26
Theta1_grad = (1/m) * (E2' * a1w1);  % 25x401

% Part 3 Regularise gradient


Theta2_reg = Theta2_grad(:, 2:(hidden_layer_size+1)) ...
           + (lambda/m) * Theta2(:, 2:(hidden_layer_size+1));
Theta1_reg = Theta1_grad(:, 2:(input_layer_size+1)) ...
           + (lambda/m) * Theta1(:, 2:(input_layer_size+1));

Theta2_grad = [Theta2_grad(:, 1) Theta2_reg];
Theta1_grad = [Theta1_grad(:, 1) Theta1_reg];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
