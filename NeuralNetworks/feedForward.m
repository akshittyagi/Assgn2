function [delta , costCalculated , a2] = feedForward(X,Y,Theta1,Theta2,k)
   
   activationForHiddenLayer = sigmoid(Theta1*X); 
   activationForHiddenLayer = [1 ; activationForHiddenLayer];
   predictedOutput = sigmoid(Theta2*activationForHiddenLayer);
   
   a2 = activationForHiddenLayer;
   
   classNo = Y;
   outputVector = zeros(k,1);
   
   if(classNo == 0)
       classNo = 10;
   end
   
   outputVector(classNo) = 1;
   
   logh = log(predictedOutput);
   log1_h = log(1-predictedOutput);
   sum = 0;
   for i=1:classNo
       sum = sum + outputVector(i)*logh(i) + (1-outputVector(i))*(log1_h);
   end
   
   delta = predictedOutput - outputVector;
   costCalculated = sum;
end