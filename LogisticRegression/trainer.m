function [thetaTrained] = trainer(X,Y,lambda,k)
    
m = size(X,1);
n = size(X,2);

X = [ones(m,1) X];


initialParams = zeros(n+1,1);
thetaTrained = zeros(k,n+1);

for i=1:k
    
    [tempTheta] = trainerForOne(X,Y,lambda,initialParams,m,n,i);
    thetaTrained(i,:) = tempTheta(:)';
end

end
