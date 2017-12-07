clear;
clc;
more off;

% Load training data
printf("Reading training data\n");
all_data = dlmread('../train.csv', ',', 1,0);

train_qty = 10000;   % Max of 42000
[images_train labels_train data_train label_train] = load_data(all_data, train_qty);

% Training
time1 = time();
[a_param cost] = train_regression(data_train, label_train);
printf("Training took %d seconds\n", time() - time1);

% Classifing
[prob, pred] = test_regression(a_param, images_train);
pred_hits = (pred == labels_train);

right = sum(pred_hits);
wrong = size(images_train)(1) - right;

accuracy = mean(double(pred_hits));

printf("Hits: %d, Miss: %d. Total: %d\n", right, wrong, right + wrong);
printf("Multi-class Logistic Regression accuracy: %1.2f%%\n\n", accuracy * 100);

% Load data

printf("Reading test data\n");
images_test = dlmread('../test.csv', ',', 1,0);
pred = zeros(size(images_test),2);

pred(:,1) = 1:size(images_test)(1);
[prob, pred(:,2)] = test_regression(a_param, images_test);

% Output file
printf("File results.csv created.\n");
dlmwrite("results.csv", pred, ',');
