function [Theta1 , Theta2] = trainNN(X,Y,lambda,k,M)

neuronsInHiddenLayer = 10;
m = size(X,1);
n = size(X,2);

X = [ones(m,1) X];

epsilon1 = sqrt(6)/sqrt( neuronsInHiddenLayer+1 + n+1 );
epsilon2 = sqrt(6)/sqrt( k + neuronsInHiddenLayer+1 );

Theta1 = (epsilon1*2).*rand(neuronsInHiddenLayer,n+1) - epsilon1;
Theta2 = (epsilon2*2).*rand(k,neuronsInHiddenLayer+1) - epsilon2;



alpha = 0.1;

maxIter = 100;

for count=1:maxIter
    
    
    Delta2 = zeros(k,neuronsInHiddenLayer+1);
    Delta1 = zeros(k,n+1);

    for i=1:m
     a1 = X(i,:)';
     [delta3 , cost , a2] = feedForward(a1,Y(i),Theta1,Theta2,k); 
     delta2 = ((Theta2')*delta3).*(a2.*(1-a2));
     delta2 = delta2(2:end);
     Delta1 = Delta1 + delta2*(a1');
     Delta2 = Delta2 + delta3*(a2');
    end
    temp1 = Delta1;
    temp2 = Delta2;
    
    Delta1(:,2:end) = (Delta1(:,2:end)/10)+(lambda/10)*Theta1(:,2:end);
    Delta2(:,2:end) = (Delta2(:,2:end)/10)+(lambda/10)*Theta2(:,2:end); 
    Delta1(:,1) = temp1(:,1)/10;
    Delta2(:,1) = temp2(:,1)/10;
    
    Theta1 = Theta1 - alpha*Delta1;
    Theta2 = Theta2 - alpha*Delta2;
    
end

% for count=1:(m/10)
%     
%     E = count*10;
%     S = E - 10;
%     for i=E:S
%      a1 = X(i,:)';
%      [delta3 , cost , a2] = feedForward(a1,Y(i),Theta1,Theta2,k); 
%      delta2 = ((Theta2')*delta3).*(a2.*(1-a2));
%      delta2 = delta2(2:end);
%      Delta1 = Delta1 + delta2*(a1');
%      Delta2 = Delta2 + delta3*(a2');
%     end
%     temp1 = Delta1;
%     temp2 = Delta2;
%     
%     Delta1(:,2:end) = (Delta1(:,2:end)/10)+(lambda/10)*Theta1(:,2:end);
%     Delta2(:,2:end) = (Delta2(:,2:end)/10)+(lambda/10)*Theta2(:,2:end); 
%     Delta1(:,1) = temp1(:,1)/10;
%     Delta2(:,1) = temp2(:,1)/10;
%     
%     Theta1 = Theta1 - alpha*Delta1;
%     Theta2 = Theta2 - alpha*Delta2;
%     
% end
 
end