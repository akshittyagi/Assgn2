function [theta] = trainerForOne(X,Y,lambda,initialParams,m,n,i)

alpha = 0.1;
maxIter = 400;

requiredY = (Y==i);    


for count=1:maxIter
   h = sigmoid(X*initialParams);
   initialParams = initialParams + alpha*(X'*(requiredY - h));
   
end
theta = initialParams;
end