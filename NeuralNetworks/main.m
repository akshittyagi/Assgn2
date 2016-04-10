clc;
clear;
close all;

X = importdata('training_images.txt');
Y = importdata('training_labels.txt');


Y(Y==0)=10
pause;
lambda = 5;
k = 10;
M = 4000;

X1 = X(1:4000,:);
Y1 = Y(1:4000);

X2 = X(4001:5000,:);
Y2 = Y(4001:5000);

neuronsInHiddenLayer = 10;
m = size(X,1);
n = size(X,2);
epsilon1 = sqrt(6)/sqrt( neuronsInHiddenLayer+1 + n+1 );
epsilon2 = sqrt(6)/sqrt( k + neuronsInHiddenLayer+1 );


initTheta1 = (epsilon1*2).*rand(neuronsInHiddenLayer,n+1) - epsilon1;
initTheta2 = (epsilon2*2).*rand(k,neuronsInHiddenLayer+1) - epsilon2;

initialParams = [initTheta1(:) ; initTheta2(:)];

options = optimset('MaxIter', 50);
costFunction = @(p) nnCostFunction(p,n,neuronsInHiddenLayer,k, X1, Y1, lambda);

[trainedTheta, costCalculated] = fmincg(costFunction, initialParams, options);

Theta1 = reshape(trainedTheta(1:neuronsInHiddenLayer * (n + 1)),neuronsInHiddenLayer, (n + 1));

Theta2 = reshape(trainedTheta((1 + (neuronsInHiddenLayer * (n + 1))):end),k, (neuronsInHiddenLayer+ 1));




predValuesForTrainingSet = predict(Theta1, Theta2, X1);
predValuesForTestingSet = predict(Theta1,Theta2,X2);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(predValuesForTrainingSet== Y1)) * 100);
fprintf('\nTesting Set Accuracy: %f\n', mean(double(predValuesForTestingSet == Y2)) * 100);