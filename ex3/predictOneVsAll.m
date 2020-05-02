function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       


%{
  When there's only logistic regression involved (no neural networks), 
  it is legal to just compute X * all_theta' and compare with y for 
  a correct prediction, This is because sigmoid function is monotonically
  increasing and flattens out at certain x value (converge to a limit) when
  x > 0 and monotonically decreasing when x< 0. 
  Therefore, the larger x is (x > 0), the larger y would be (vice versa for
  x < 0 ). Accordingly, we can tell from just x rather than calculating y.
  
  However, neural network is slightly different in that the values that 
  are forward propagated are the y (the actual g(z)) rather than x(z).
  It is wrong to just calculate X * all_theta(Z) and leave out the actual
  g(Z), i.e. y.)
}%

gz = sigmoid(X * all_theta');
[g, pred] = max(gz, [], 2); % get largest items in each row (dim 2)
p = pred;

% =========================================================================

end
