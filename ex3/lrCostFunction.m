function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
thetax = X * theta;
hx = sigmoid(thetax);

cost_y_one = (-y)' * log(hx);
cost_y_zero = (1-y)' * log(1 - hx);
sum_cost = cost_y_one - cost_y_zero;
J = (1/m) * sum_cost;

error = hx - y;
g = (1/m) * (X' * error); %unregularised gradient

cost_reg_term = (lambda / (2*m)) * (theta(2:end)' * theta(2:end));
J = J  + cost_reg_term;


thetax = X * theta;
hx = sigmoid(thetax);
error = hx - y;

grad0 = g(1);
grad_rest = (1/m) * (X(:, 2:end)' * error);
grad_reg_term = (lambda / m) * theta(2:end);
grad_rest = grad_rest + grad_reg_term;
grad = [grad0; grad_rest];

% =============================================================
grad = grad(:);
end
