function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% size(theta) == (2,1)
% size(X) == (12,2)

% size(X*theta) = (12,1)
% size(y) = (12, 1)

% size(X*theta - y)  = (12, 1)

% size((X*theta - y).^2) = (12, 1)
% (X * theta - y)
% (X * theta - y).^2
Jn = (1/(2*m)) * (sum( (X * theta - y).^2)) ;
t = theta'(2:end);
R = (lambda/(2*m)) * sum(t * t');
J = Jn + R;


grad(1, 1) = (1/m) * ones(1, m) * ((X*theta - y).* X(:, 1) );
for i = 2:size(theta)(1)
	grad(i, 1) = (1/m) * ones(1, m) * ((X*theta - y).* X(:, i) ) + (lambda/m) * theta(i, 1);
endfor





% =========================================================================

grad = grad(:);

end
