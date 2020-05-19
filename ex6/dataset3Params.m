function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_test = [.01 .03 .1 .3 1 3 10 30];
sigma_test = [.01 .03 .1 .3 1 3 10 30];
[p,q] = meshgrid(c_test, sigma_test);
pairs = [p(:) q(:)];  % all possible pairs from c_test and sigma_test
error_min = inf;
c_best = inf;
s_best = inf;

for i = 1:length(pairs)
    pair_cell = num2cell(pairs(i,:));
    [c, s] = pair_cell{:};
    
    % train with current C and sigma
    model_cv = svmTrain(X, y, c, ...
                       @(x1, x2) gaussianKernel(x1, x2, s));
                   
    pred_cv = svmPredict(model_cv, Xval);
    
    error_cv = mean(double(pred_cv ~= yval));
    
    if error_cv < error_min
        error_min = error_cv;
        c_best = c;
        s_best = s;
        fprintf("current error: %.8f \n", error_min)
        fprintf("best c: %.4f, best sigma: %.4f", c_best, s_best)
    end
    fprintf("current loop: %f", i)
end

C = c_best;
sigma = s_best;
    

% =========================================================================

end
