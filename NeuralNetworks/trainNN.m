function [Theta1 , Theta2] = trainNN(X,Y,lambda,k)

neuronsInHiddenLayer = 10;
m = size(X,1);
n = size(X,2);

X = [ones(m,1) X];

epsilon1 = sqrt(6)/sqrt( neuronsInHiddenLayer+1 + n+1 );
epsilon2 = sqrt(6)/sqrt( k + neuronsInHiddenLayer+1 );

Theta1 = (epsilon1*2).*rand(neuronsInHiddenLayer,n+1) - epsilon1;
Theta2 = (epsilon2*2).*rand(k,neuronsInHiddenLayer+1) - epsilon2;

sum = 0;

for i=1:m
    sum = sum + feedForward(X(i,:)',Y(i),Theta1,Theta2,k,neuronsInHiddenLayer);
end

cost = (-sum)/m;
 
end