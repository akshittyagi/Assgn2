X = importdata('training_images.txt');
Y = importdata('training_labels.txt');

lambda = 1;
k = 10;

[Theta1 , Theta2] = trainNN(X,Y,lambda,k);