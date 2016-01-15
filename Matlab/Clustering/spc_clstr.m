%%  Dirver function for Spectral Clustering
function spc_clstr(sigma)
    input = circs;
    input = input';

    A = similarity(input, sigma);
    C = basic_alg(A, 2);
    disp(C);
    color = zeros(size(input,1), 3);
    for i = 1 : size(input, 1)
        if C(i) == 1
            color(i, :) = [0 0.8 1];
        else
            color(i, :) = [1 0 0];
        end
    end
    scatter(input(:, 1), input(:, 2), 30 ,color, 'filled');
end