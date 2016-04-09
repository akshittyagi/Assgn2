clear;
close all;
clc;

X = importdata('training_images.txt');
Y = importdata('training_labels.txt');

y = Y(1:3000);
x = X(1:3000,:);

x1 = X(3001:5000,:);
y1 = Y(3001:5000);

k = 10;
lambda = 1;

m = size(X,1);
randomIndices = randperm(m);
RandomDataPoints = X(randomIndices(1:100),:);

displayData(RandomDataPoints);

[thetaTrained] = trainer(x,y,lambda,k);

predictedValuesForTesting = predictOneVsAll(thetaTrained, x1);
predictedValuesForTraining = predictOneVsAll(thetaTrained, x);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(predictedValuesForTesting == y1)) * 100);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictedValuesForTraining == y)) * 100);