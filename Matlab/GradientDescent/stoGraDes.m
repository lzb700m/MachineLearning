% Stochastic gradient descent function
function [w, b, iter] = stoGraDes(X, Y, w_init, b_init, rate)
[numRow, numCol] = size(X);
w = w_init;
b = b_init;
i = 0;
iter = 0;
% counter is used to track the number of correctly classified data point,
% when it reaches 100, all data point are correctly classified. If a
% misclassification is encountered, counter is reset to zero
counter = 0;
while 1
    display(iter);
    display(w);
    display(b);
    display('-----------------------------------------------------------');
    i = i + 1;
    iter = iter + 1;
    
    % if reaches all n data point, start back at beginning
    if (i > numRow)
        i = rem(i, numRow);
    end
    
    % update w and b if data sample i is misclassified, otherwise
    % increament correctly classified counter
    if ((-Y(i) * (w * X(i, :)' + b)) >= 0)
        w = w - rate * (-Y(i) * X(i, :));
        b = b - rate * (-Y(i));
        counter = 0;
    else
        counter = counter + 1;        
    end
    
    % looping end condition (counter >= 100)
    if (counter == numRow)
        break;
    end
end
end
