%Initialization
%======================================================================
clear;close all;clc;
data = load('econdata.txt');
X = data(:,[1,2,3,4]);
y = data(:,5);
[m,n] = size(X);
X = [ones(m,1) X];
initial_theta = zeros(n+1,1);
%======================================================================
%
%Training
%======================================================================
lambda=1;
[cost, grad] = costFunction(initial_theta, X, y,lambda);
options = optimset('GradObj','on','MaxIter','500');
[theta,cost] = ...
    fminunc(@(t)(costFunction(t,X,y,lambda)),initial_theta,options);
%=======================================================================
%
%Predicting
%=======================================================================
res=load('hypodata.txt');
[l,b]=size(res);
res=[ones(l,1) res];
p = predict(theta,res);
for i = 1: size(p),
  if(p(i)==1)
    fprintf("REPUBLICAN \n");
   else
    fprintf("DEMOCRAT \n");
    endif
endfor
%========================================================================
