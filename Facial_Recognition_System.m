% Dhruv Meena
% Question 4
IMAGE_S = imread("ORL\s1\1.pgm","pgm"); % get a sample image to get the size that we are working with
[IMAGE_HEIGHT, IMAGE_WIDTH] = size(IMAGE_S); % we got the size

PEOPLE = 32;
IMAGES_TRAIN_N = 6;
k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 170];

TOTAL_IMAGES = PEOPLE*IMAGES_TRAIN_N; % width of the data matrix, height being an image flattened as column vector

% ORL dataset training data generated
TRAINING_DATA = GENERATE_TRAINING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH, TOTAL_IMAGES);

% ORL dataset testing data generated
TESTING_DATA = GENERATE_TESTING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH);

% ORL dataset eigenfaces generated
[EIGEN_FACES, DATA_C, EIGEN_VECTORS] = PCA(TRAINING_DATA);

% Perform facial recognition on ORL dataset and plot k values vs
% recognition rates
FACIAL_RECOGNITION(DATA_C, EIGEN_FACES, k_values, TESTING_DATA);

% Reconstructing image with various k-values
% Load the image to be reconstructed (for example, "s1/1.pgm")
sample_image = imread("ORL\s1\1.pgm", "pgm");
[IMAGE_HEIGHT, IMAGE_WIDTH] = size(sample_image);
flattened_image = double(sample_image(:)); % Flatten the image to a column vector

k_values = [2, 10, 20, 50, 75, 100, 125, 150, 175];

figure;
for idx = 1:length(k_values)
    k = k_values(idx);
    
    % Project the image onto the first k eigenfaces
    projection = EIGEN_FACES(:, 1:k)' * (flattened_image - mean(TRAINING_DATA, 2));
    
    % Reconstruct the image
    reconstruction = EIGEN_FACES(:, 1:k) * projection + mean(TRAINING_DATA, 2);
    reconstructed_image = (reshape(reconstruction, [IMAGE_HEIGHT, IMAGE_WIDTH]));
    
    % Display the reconstructed image
    subplot(3, 3, idx); % Adjust the subplot grid size as needed
    imshow(uint8(reconstructed_image));
    title(['Reconstructed with k = ', num2str(k)]);
end



function TRAINING_DATA = GENERATE_TRAINING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH, TOTAL_IMAGES)
    TRAINING_DATA = zeros(IMAGE_HEIGHT * IMAGE_WIDTH, TOTAL_IMAGES); % preallocating memory for this massive matrix
    for i = 1:32    
        for j = 1:6    
            img = imread(fullfile("ORL\", "s" + num2str(i), [num2str(j), '.pgm'])); % reading each image
            TRAINING_DATA(:, (i-1)*6 + j) = img(:);  % Flatten the image
        end
    end
end

function TESTING_DATA = GENERATE_TESTING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH)
    TESTING_DATA = zeros(IMAGE_HEIGHT * IMAGE_WIDTH, 32*4); % preallocating memory for this massive matrix
    for i = 1:32    
        for j = 1:4    
            img = imread(fullfile("ORL\", "s" + num2str(i), [num2str(j+6), '.pgm'])); % reading each image
            TESTING_DATA(:, (i-1)*4 + j) = img(:);  % Flatten the image
        end
    end
end

function [EIGEN_FACES, DATA_C, EIGEN_VECTORS] = PCA(DATA)
    % compute mean
    MEAN_DATA = mean(DATA,2);
    DATA_C = DATA - MEAN_DATA;
    COVARIANCE_MATRIX = (DATA_C.' * DATA_C)/(size(DATA, 2) - 1);
    [EIGEN_VECTORS, ~] = eigs(COVARIANCE_MATRIX,(size(DATA, 2) - 1));
    EIGEN_FACES = DATA_C * EIGEN_VECTORS;
    for i = 1:size(EIGEN_FACES, 2)
        EIGEN_FACES(:, i) = EIGEN_FACES(:, i) / norm(EIGEN_FACES(:, i));
    end
end

function [] = FACIAL_RECOGNITION(DATA_C, EIGEN_FACES, k_values, TESTING_DATA)
    
    MEAN_TEST_DATA = mean(TESTING_DATA,2);
    DATA_TEST_C = TESTING_DATA - MEAN_TEST_DATA;
    
    RATES = zeros(length(k_values), 1);
    
    for k = k_values
        TRAIN_PROJECTED = EIGEN_FACES(:, 1:k).' * DATA_C;
        CORRECT = 0;
        for i = 1:32
            for j = 1:4
                TEST_IMG = DATA_TEST_C(:, (i-1)*4 + j);
                TEST_PROJ = EIGEN_FACES(:, 1:k).' * TEST_IMG;
                
                % Calculate the distance to all training projections
                DISTANCES = sum((TRAIN_PROJECTED - TEST_PROJ).^2, 1); % SUM OF SQUARED DIFFERENCES
                [~, min_idx] = min(DISTANCES); % INDEX OF IMAGE WITH MINIMUM DIFFERENCE
                
                if ceil(min_idx / 6) == i % FIND THE TRAINING IMAGE INDEX, IF IT MATCHES WITH TEST INDEX, ADD 1 TO CORRECT
                    CORRECT = CORRECT + 1;
                end
            end
        end
        RATES(k == k_values) = CORRECT / (32 * 4); % NUM OF CORRECT/TOTAL TESTING IMAGES = RECOGNITION RATE
    end
    plot(k_values, RATES); % PLOT THE RATES
end