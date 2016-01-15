%%  Basic Algorithm
%   A - similarity matrix of sample data set
%   k - # of clusters
%   C - clustering output
%   V - n by k matrix whose columns are the eigenvectors that correspond to the k
%   smallest eigenvalues of L
%   C - final clustering
function C = basic_alg(A, k)
    n = size(A, 1);
    L = lapMtr(A);
    V = zeros(n, k);
    [EV, ED] = eig(L);
    for i = 1 : k
        [m, index] = min(diag(ED));
        V(:, i) = EV(:, index);
        ED(index, index) = inf;
    end
    % cluster V using kmeans()
    S = kmeans(V, k);
    C = S;
end

%%  Calculate Laplacian Matrix
%   A - input similarity matrix
%   D - degree matrix of A (diagnal matrix)
%   L - Laplacian matrix
function L = lapMtr(A)
    n = size(A, 1);
    D = zeros(n, n);
    for i = 1 : n
        sumAi = 0;
        for j = 1 : n
            sumAi = sumAi + A(i, j);
        end
        D(i, i) = sumAi;
    end
    L = D - A;
end