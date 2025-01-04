% Dhruv Meena
% Question 4
IMAGE_S = imread("CroppedYale\yaleB01\yaleB01_P00A+000E+00.pgm","pgm"); % get a sample image to get the size that we are working with
[IMAGE_HEIGHT, IMAGE_WIDTH] = size(IMAGE_S); % we got the size

PEOPLE = 38;
IMAGES_TRAIN_N = 40;
k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];

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

function TRAINING_DATA = GENERATE_TRAINING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH, TOTAL_IMAGES)
    TRAINING_DATA = zeros(IMAGE_HEIGHT * IMAGE_WIDTH, TOTAL_IMAGES); % preallocating memory for this massive matrix
    for i = 1:39 
        if i == 14
            continue
        end
        formatted_number = sprintf('%02d', i);
        dir_path = append("CroppedYale/", "yaleB", formatted_number, "/");
        image_paths = dir(fullfile(dir_path, '*.pgm'));
        for j = 1:40
            if ~image_paths(j).isdir % Check if it's not a directory
                full_path = append(dir_path, image_paths(j).name);
                img = imread(full_path,"pgm"); % reading each image
                TRAINING_DATA(:, (i-1)*40 + j) = img(:);  % Flatten the image
            end
        end
    end
end

function TESTING_DATA = GENERATE_TESTING_DATA(IMAGE_HEIGHT, IMAGE_WIDTH)
    TESTING_DATA = zeros(IMAGE_HEIGHT * IMAGE_WIDTH, 38*24); % preallocating memory for this massive matrix
    for i = 1:39
        if i == 14
            continue
        end
        formatted_number = sprintf('%02d', i);
        dir_path = append("CroppedYale/", "yaleB", formatted_number, "/");
        image_paths = dir(fullfile(dir_path, '*.pgm'));
        for j = 1:numel(image_paths)-40
            if ~image_paths(j).isdir % Check if it's not a directory
                full_path = append(dir_path, image_paths(j).name);
                img = imread(full_path,"pgm"); % reading each image
                TESTING_DATA(:, (i-1)*(numel(image_paths)-40) + j) = img(:);  % Flatten the image
            end
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
    % Normalize eigenfaces
    for i = 1:size(EIGEN_FACES, 2)
        EIGEN_FACES(:, i) = EIGEN_FACES(:, i) / norm(EIGEN_FACES(:, i));
    end
end

function [] = FACIAL_RECOGNITION(DATA_C, EIGEN_FACES, k_values, TESTING_DATA)
    
    MEAN_TEST_DATA = mean(TESTING_DATA,2);
    DATA_TEST_C = TESTING_DATA - MEAN_TEST_DATA;
    
    RATES = zeros(length(k_values), 1);
    
    for k = k_values
        TRAIN_PROJECTED = EIGEN_FACES(:, 4:k+3).' * DATA_C;
        CORRECT = 0;

        for i = 1:39
            if i == 14
                continue
            end
            formatted_number = sprintf('%02d', i);
            dir_path = append("CroppedYale/", "yaleB", formatted_number, "/");
            image_paths = dir(fullfile(dir_path, '*.pgm'));
            for j = 1:numel(image_paths)-40
                TEST_IMG = DATA_TEST_C(:, (i-1)*(numel(image_paths)-40) + j);
                TEST_PROJ = EIGEN_FACES(:, 4:k+3).' * TEST_IMG;
                
                % Calculate the distance to all training projections
                DISTANCES = sum((TRAIN_PROJECTED - TEST_PROJ).^2, 1); % SUM OF SQUARED DIFFERENCES
                [~, min_idx] = min(DISTANCES); % INDEX OF IMAGE WITH MINIMUM DIFFERENCE
                
                if ceil(min_idx / 40) == i % FIND THE TRAINING IMAGE INDEX, IF IT MATCHES WITH TEST INDEX, ADD 1 TO CORRECT
                    CORRECT = CORRECT + 1;
                end
            end
        end
        RATES(k == k_values) = CORRECT / (38 * 24); % NUM OF CORRECT/TOTAL TESTING IMAGES = RECOGNITION RATE
    end
    plot(k_values, RATES); % PLOT THE RATES
end