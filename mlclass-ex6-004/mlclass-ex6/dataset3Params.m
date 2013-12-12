function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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


optional_cs = [0.01 0.03 0.1 0.3 1 3 10 30];
optional_sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];

minimum_c = optional_cs(1);
minimum_sigma = optional_sigmas(1);
current_model = svmTrain(X, y, minimum_c, @(x1, x2) gaussianKernel(x1, x2, sigma));

% Calculating error on first pair of (c, sigma) == (0.01, 0.01). We use the validation dataset.

minimum_predictions = svmPredict(current_model, Xval);
minimum_error = mean(double(minimum_predictions ~= yval));

% Looping through all other options of (c, sigma) and finding the minimum error, 
% and this way getting the best pair of (c, sigma)

for i = 1:8
	for j = 1:8

		c = optional_cs(i);
		sigma = optional_sigmas(j);
		[c, sigma]
		current_model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma));

		current_predictions = svmPredict(current_model, Xval);
		current_error = mean(double(current_predictions ~= yval))
		if current_error < minimum_error
			minimum_error = current_error;
			minimum_predictions = current_predictions;
			minimum_c = c;
			minimum_sigma = sigma;
		endif
	endfor
endfor

C = minimum_c
sigma = minimum_sigma


% =========================================================================

end
