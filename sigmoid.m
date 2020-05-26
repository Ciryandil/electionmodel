function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

%Initialization
%===================
g = zeros(size(z));
%===================
%Calculation
%=======================
g = 1./(1+exp(-z));

% ======================

end