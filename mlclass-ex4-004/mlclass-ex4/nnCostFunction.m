function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% p = predict(Theta1, Theta2, X);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

cost = 0;
% the first loop - going over all of the example set and summing.
for i = 1:m
	% converting y(1) to a binary vector representing the answer
	% yvec should be something similar to: [0 0 0 1 0] representing the number 4 out of 5 options
	yvec = index_vec(num_labels, y(i), 1);
  prediction_vec = h2(i,:);
  for k = 1:num_labels
  	cost += yvec(k) * log(prediction_vec(k)) + (1 - yvec(k)) * (log(1 - prediction_vec(k)));
  endfor
endfor

% Before regularization
J = (-1/m) * cost;

% initilizing regularization variable
regularization = 0;

% summing all Theta1 elements except the first column (the bias unit)
r_theta1 = sum(Theta1(:, 2:end)(:).^2);

% summing all Theta2 elements except the first column (the bias unit)
r_theta2 = sum(Theta2(:, 2:end)(:).^2);

regularization = r_theta1 + r_theta2;

J = J + (lambda/(2*m)) * regularization

% 
capital_delta_2 = zeros(size(Theta2)(1), size(Theta2)(2));
capital_delta_1 = zeros(size(Theta1)(1), size(Theta1)(2));

% Starting to calculate div

for t = 1:m 

	a1 = [1 X(t,:)];
	% Calculating the second layer activation values.
	% The result is a2 which is a 1*25 vector. 
	% Each column (i) represent the the value for the i-th activation unit in the second layer.

	z2 = [1 X(t,:)] * Theta1';
	a2 = sigmoid(z2);

	% Calculating the third layer activation values.
	% The result is a3 which is a 1*26 vector;
	% Each column represent the value for the i-th activation unit in the third layer.

	z3 = [1 a2] * Theta2';
	a3 = sigmoid(z3);

	% How far is our prediction on the t-th example in X, from the actual class of it, 
	% given by the y vector.
	y_vec = index_vec(num_labels, y(t), 1);
	delta_3 = a3' - y_vec';

	% Hidden layer delta
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z2']);


	% Calculating capital deltas (the sum of errors for every parameter in Theta1/Theta2)

	
	capital_delta_1 = capital_delta_1 + delta_2(2:end) * a1;
	
	a2 = [1 a2];
	capital_delta_2 = capital_delta_2 + delta_3 * a2;

endfor

Theta1_copy = [zeros(size(Theta1)(1), 1), Theta1(:, 2:end)];
Theta2_copy = [zeros(size(Theta2)(1), 1), Theta2(:, 2:end)];

Theta1_grad = (1/m) * capital_delta_1 + lambda/m * Theta1_copy;

Theta2_grad = (1/m) * capital_delta_2 + lambda/m * Theta2_copy;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
