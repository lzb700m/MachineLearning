%Dual SVM using Gaussian Kernel
%Input:  data X 
%slack penalty constant C
%standard deviation sigma
%Output:  the vector lambda concactenated with the scalar b
function [lambda,b] = dualSVM_train(X,C,sigma)
[r,c] = size(X);
x = X(:,1:end-1);
y = X(:,c);y(y==0) = -1;

%Gaussian kernel
kernel=zeros(r,r);
H=zeros(r,r);
for i=1:r
    for j=1:r
        kernel(i,j) = exp(-norm(x(i,:)-x(j,:))^2/(2*sigma^2));
        H(i,j) = y(i)*y(j)*kernel(i,j);
    end
end

f=-ones(r,1);
Aeq = y';
beq = 0;
lb = zeros(r,1);
ub = C*ones(r,1);
opts = optimset('Algorithm','interior-point-convex');
[lambda]= quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts); 

wTx=zeros(1,r);
for j=1:r
    wTx(j) = (lambda.*y)'*kernel(:,j);
end
B = y - wTx';
b = mean(B); 


