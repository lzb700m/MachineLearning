%%  Create similarity matrix of given data set
%   input - data set for clustering, every sample is one row in input
%   A - similarity matrix
%   sigma - parameter for calculating similarity
function A = similarity(input, sigma)
    n = size(input, 1); % # of samples
    A = zeros(n, n);
    for i = 1 : n
        for j = i : n
            diff = input(i, :) - input(j, :);
            A(i, j) = exp(-(diff * diff') / (2 * sigma * sigma));
            A(j, i) = A(i, j);           
        end
    end
end