function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(X); % number of training examples, then shouldn't be y.
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% 12 x 2

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute unregularised cost and gradient;

hx = X * theta;
sigma = hx - y;
sigma = sigma.^2;   % or sigma' * sigma
error = sum(sigma);
J = 1/(2*m) * error;

error = hx - y;
g = (1/m) * (X' * error);

% Compuate regularised term

cost_reg_term = (lambda / (2*m)) * (theta(2:end)' * theta(2:end));
J = J  + cost_reg_term;

grad0 = g(1);
grad_rest = (1/m) * (X(:, 2:end)' * error);
grad_reg_term = (lambda / m) * theta(2:end);
grad_rest = grad_rest + grad_reg_term;
grad = [grad0; grad_rest];


% =========================================================================

grad = grad(:);

end
