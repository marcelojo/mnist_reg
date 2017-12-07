function [a_param J] = train_regression (data, label)

data = (data .- min(data)) ./ (max(data) .- min(data) + 1);


% Define max interation
max_iteration = 50;

% Define learning rate step
learning_rate = 0.6;

% Define regularization
lambda = 450;

% Initialize parameters for all 10 classes
a_param = zeros(10, size(data)(2) + 1);
J = zeros(10, max_iteration);
%a_param = dlmread('a_param.csv', ',', 0,0);

% Loop for training
printf("Training: dataset %d images, iteration = %d, alpha = %1.2f, lambda = %1.2f\n\n", size(data)(1), max_iteration, learning_rate, lambda);
for class = 0:9
    for i = 1:max_iteration
        [a_param(class + 1, :) J(class + 1, i)] = step_gradient(a_param(class + 1,:), data, label == class, learning_rate, lambda);
        printf("Class %d iterating: %d of %d \r", class, i, max_iteration);
        fflush(stdout);
    end
    printf("\n");
    fflush(stdout);
end
endfunction