%%  Image clutering using spectral clustering algorithm
%   inputFile - input file name
%   sigma - parameter for spectral clustering
%   k - number of clusters
function C = spc_img(inputFile, sigma, k)
    inputImg = imread(inputFile);
    [rows, cols] = size(inputImg);
    inputVec = importImg(inputFile);
    A = similarity(inputVec, sigma);
    cVec = basic_alg(A, k);
    C = reshape(cVec, rows, cols);
    pic = mat2gray(C, [1, 2]);
    imwrite(pic, 'output.jpg');
end

%%	Import an image file and put the pixel gray value into a vector
%   inputFile - filename of the image
function V = importImg(inputFile)
    temp = imread(inputFile);
    [rows, cols] = size(temp);
    V = zeros(rows * cols, 1);
    for i = 1 : cols
        for j = 1 : rows
            V((i - 1) * rows + j) = temp(j, i);
        end
    end
end