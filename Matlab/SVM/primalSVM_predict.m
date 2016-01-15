%Primal SVM predict
%Input:  data X
%separator weight vector wts 
%scalar b
%Output: accuracy

function [accuracy] = primalSVM_predict(X, wts, b)
[r,c]=size(X);
count=0;
y = X(:,c); y(y==0) = -1;
for j=1:r
    f_x = dot(X(j,1:c-1),wts) + b;
    if(f_x * y(j) > 0)
        count = count+1;
    end
end
accuracy = count/r*100;
disp(['accuracy = ',num2str(accuracy)]);