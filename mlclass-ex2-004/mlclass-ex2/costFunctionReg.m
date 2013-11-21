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

y2 = [y 1-y];
tx = X * theta;
htx = sigmoid(tx);

htx_with_minus = [htx'; (1 - htx)'];
log_htx_with_minux = log(htx_with_minus);
J = (-1/m) *((diag(y2 * log_htx_with_minux))' * ones( m, 1)) + (lambda/(2*m)) * ( theta'(1,2:end).^2 * ones(size(theta)(1) - 1, 1) );


htx_minus_y = htx - y;
theta2 = [0 ((lambda/m) * theta'(1, 2:end))];

grad = ((1/m) * htx_minus_y' * X) + theta2;

% =============================================================

end
