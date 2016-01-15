% Standard gradient descent function
function [w, b, iter] = stdGraDes(X, Y, w_init, b_init, rate)
[numRow, numCol] = size(X);
w = w_init;
b = b_init;
iter = 0;
while 1
    delta_w = [0 0 0 0];
    delta_b = 0;
    display(iter);
    display(w);
    display(b);
    display('-----------------------------------------------------------');
    % calculate accumulative loss gradient
    for i = 1:numRow
       if ((-Y(i) * (w * X(i, :)' + b)) >= 0)
           delta_w = delta_w + (-Y(i) * X(i, :));
           delta_b = delta_b + (-Y(i));
       end
    end
    % condition to terminate the loop
    if (delta_w == 0) & (delta_b == 0)
        break;
    end
    w = w - rate * delta_w;
    b = b - rate * delta_b;
    iter = iter + 1;
end
end
