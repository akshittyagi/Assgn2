function [theta] = trainerForOne(X,Y,lambda,initialParams,m,n,i)

alpha = 0.1;
maxIter = 400;

requiredY = (Y==i);    

old = lrCostFunction(initialParams,X,Y,lambda);

for count=1:maxIter
   if(lrCostFunction(initialParams,X,Y,lambda)>old)
       count = maxIter;
   end
   h = sigmoid(X*initialParams);
   initialParams = initialParams + alpha*(X'*(requiredY - h));
   
   old = lrCostFunction(initialParams,X,Y,lambda);
end
theta = initialParams;
end