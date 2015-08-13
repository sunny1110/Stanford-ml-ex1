function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
h = 0; % Hypothesis Value
eta =0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	     h = X * theta; % h is the hypothesis.
	     theta = theta - transpose(alpha * (transpose(h-y) * X) ./ m);
%A brief description of the above equation(s):
%h stores the hypothesis, therefore h-y gives us the error. h-y is sized 97 X 1, X is sized 97 X 2
%What we need is the sum of the error times element product X. Basically each element of column of error is to be multipled by the corresponding
%Column element of X, thereby yielding a 97 X 2 matrix, for which the sum is to be taken for each row. However as per matrix multiplication
  %Rules tranpose(X) * X gives us this result, escaping any need for the use of sum function. Multiply and divide by alpha and m respectively
  %Apply the transpose again to suit thetas dimensions. Take difference. Problem solved, BINGO! Or as Al Pacino would say HO-AH!
	    



	     


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
