%%  Diver function for kmeans clustering
function vkmean()
    input = circs;
    input = input';
    [cluster, centroid] = kmeans(input, 2);
    color = zeros(size(input,1), 3);
    for i = 1 : size(input, 1)
        if cluster(i) == 1
            color(i, :) = [0 0.8 1];
        else
            color(i, :) = [1 0 0];
        end
    end
    scatter(input(:, 1), input(:, 2), 30 ,color, 'filled');
    % plot centroid
    hold on;
        plot(centroid(:, 1), centroid(:, 2), '+k', 'MarkerSize', 30);
    hold off;
end