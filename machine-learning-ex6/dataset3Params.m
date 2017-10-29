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


% C_vec = [];
% sigma_vec = [];
% error_vec = [];
% 
% fprintf('--------------\n');
% fprintf('Start searching for optimal C and sigma values\n');
% 
% try_values = [0.01 0.03 0.1 0.3 1 3 10 30];
% for C_try = try_values
%     for sigma_try = try_values
%         
%         fprintf('Train/validate (cross validation set) for [C,sigma] = [%f, %f]\n', C_try, sigma_try);
%         
%         model = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
%         predictions = svmPredict(model, Xval);
%         err = mean(double(predictions ~= yval));
%         
%         C_vec = [C_vec; C_try];
%         sigma_vec = [sigma_vec; sigma_try];
%         error_vec = [error_vec; err];
%         
% 
%     end
% end
% 
% [rdn, index] = min(error_vec);
% 
% 
% C = C_vec(index)
% sigma = sigma_vec(index)

C = 1;
sigma = 0.1;

% =========================================================================

end
