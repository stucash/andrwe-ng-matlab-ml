function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[cost, g] = costFunction(theta, X, y);
cost_reg_term = (lambda / (2*m)) * (theta(2:end)' * theta(2:end));
J = cost  + cost_reg_term;

thetax = X * theta;
hx = sigmoid(thetax);
error = hx - y;

grad0 = g(1);
grad_rest = (1/m) * (X(:, 2:end)' * error);
grad_reg_term = (lambda / m) * theta(2:end);
grad_rest = grad_rest + grad_reg_term;
grad = [grad0; grad_rest];

% =============================================================

end
