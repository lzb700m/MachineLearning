%Dual SVM predict using Gaussian Kernel
%Input: train data X 
%train/validation/test data Y
%standard deviation sigma
%the vector lambda lambda
%bias term b
%Output: accuracy
function [accuracy] = dualSVM_predict(X, Y, sigma, lambda, b)
[r,c] = size(X);
y = X(:,c);
y(y==0) = -1;
x = X(:,1:end-1);

[r1,c1] = size(Y);
y1 = Y(:,c1);
y1(y1==0) = -1;
yx = Y(:,1:end-1);

%Gaussian kernel
kernel=zeros(r1,r);
wTx = zeros(r1,1);
for i=1:r1
    for j=1:r
        kernel(j,i) = exp(-norm(x(j,:)-yx(i,:))^2/(2*sigma^2));
        wTx(i) = wTx(i) + lambda(j)*y(j)*kernel(j,i);
    end
end

pred_y = (wTx+b)>0;
pred_y = double(pred_y);
pred_y(sign(pred_y)==0) = -1;
accuracy = sum(pred_y==y1)*100/r1;
disp(['sigma = ',num2str(sigma),'; accuracy = ',num2str(accuracy)]);