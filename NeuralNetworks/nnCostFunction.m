function [J grad] = nnCostFunction(currParams, n,neuronsInHiddenLayer, k, X, y, lambda)

Theta1 = reshape(currParams(1:neuronsInHiddenLayer * (n + 1)), neuronsInHiddenLayer, (n + 1));

Theta2 = reshape(currParams((1 + (neuronsInHiddenLayer * (n + 1))):end),k, (neuronsInHiddenLayer + 1));

m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
s=0;
    
for i=1:m
    yi=zeros(k,1);
    temp=[1 X(i,:)];
    a2=sigmoid(Theta1*temp');
    a2=[1 a2'];
    a2=a2';
    a3=sigmoid(Theta2*a2);
    h=a3;
    
    num=y(i);
    
    yi(num)=1;
    
    S1=-sum(yi.*log(h));
    S2=-sum((1-yi).*log(1-h));
    s=s+S1+S2;
end

J=J+s/m;

temp1=Theta1.^2;
temp2=Theta2.^2;

s1=0;
s2=0;

for i=1:size(temp1,1)
    for j=2:size(temp1,2)
        s1=s1+temp1(i,j);
    end
end

for i=1:size(temp2,1)
    for j=2:size(temp2,2)
        s2=s2+temp2(i,j);
    end
end

J=J+(lambda/(2*m))*(s1+s2);

% -------------------------------------------------------------
%%%%%%%%%%%%% J computed %%%%%%%%%%%
%%%%%%%%%%%%% Computing Grads %%%%%
cDelta1=zeros(size(Theta1));
cDelta2=zeros(size(Theta2));
for i=1:m
    a1=X(i,:);
    a1=[1 a1];
    z2=(Theta1*a1');
    a2=sigmoid(z2);
    a2=[1 a2'];
    z3=Theta2*a2';
    a3=sigmoid(z3);
    
    yi=zeros(k,1);
    num=y(i);
    yi(num)=1;
    delta3=a3-yi;
    tempTheta2=Theta2(:,2:end);
    delta2=((tempTheta2)'*delta3).*sigmoidGradient(z2);
    %
    cDelta1=cDelta1+delta2*(a1);
    
    cDelta2=cDelta2+delta3*(a2);
end

Theta1_grad=cDelta1/m;
Theta2_grad=cDelta2/m;

t1=Theta1;
t2=Theta2;

for i=1:size(t1,1)
    t1(i,1)=0;
end

for i=1:size(t2,1)
    t2(i,1)=0;
end

t1=t1*lambda/m;
t2=t2*lambda/m;
Theta1_grad=Theta1_grad+t1;
Theta2_grad=Theta2_grad+t2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
