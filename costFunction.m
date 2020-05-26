function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialization
%=======================================================
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
%=======================================================

%Calculations
%=======================================================
rtheta = theta(2:end);
h = sigmoid(X*theta);
balance= (lambda/(2*m))*(sum(rtheta.*rtheta));
gbalance = (lambda/m)*rtheta;
J = sum(-y.*(log(h))-(1-y).*(log(1-h)))/m+balance; 
grad = (X'*(h-y))/m;
grad(2:end)=grad(2:end)+gbalance;
% =============================================================

end
