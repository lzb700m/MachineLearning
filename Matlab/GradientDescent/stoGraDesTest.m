% clear work space and initialize X, Y, w, b
clear;
load perceptron.data;
X = perceptron(:, 1:4);
Y = perceptron(:, end);
w_init = [0 0 0 0];
b_init = 0;
rate = 1;
diary on;
diary('problem2_output.txt');
% perform stochastic gradient descent
[w, b, iter] = stoGraDes(X, Y, w_init, b_init, rate);
diary off;