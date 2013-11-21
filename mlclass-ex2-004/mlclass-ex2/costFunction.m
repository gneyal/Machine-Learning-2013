function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

y2 = [y 1-y];
tx = X * theta;
htx = sigmoid(tx);

htx_with_minus = [htx'; (1 - htx)'];
log_htx_with_minux = log(htx_with_minus);
J = (-1/m) *((diag(y2 * log_htx_with_minux))' * ones( m, 1));


htx_minus_y = htx - y;
grad = (1/m) * htx_minus_y' * X;

% =============================================================

end
