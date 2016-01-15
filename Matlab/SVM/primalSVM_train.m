%Primal SVM  Variables X=[w b e]'
%Objective function: min 0.5*(XHX')+f'X
%Input:  data X 
%slack penalty constant C
%Output:  the vector w concactenated with the scalar b
function [wts,b] = primalSVM_train(X, C)
[r,c] = size(X);
%y = X(:,c);%class labels
y = X(:,c); y(y==0) = -1;
H = diag([ones(1,c-1),zeros(1,r+1)]);
f= [zeros(c,1); C*ones(r,1)];
%inequality: -yi(w'x+b)-ei<=-1
A = -1*[X(:,1:c-1).*(y*ones(1,c-1)) y eye(r)];
b = -1*ones(r,1);
%ei>=0
lb = [-inf*ones(c,1); zeros(r,1)];
opts = optimset('Algorithm','interior-point-convex');
[w] = quadprog(H,f,A,b,[],[],lb,[],[],opts);
wts = w(1:c-1,1);
b=w(c,1);




