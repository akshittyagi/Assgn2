function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

for i=1:m
    a1 = [1 X(i,:)];
    a2 = sigmoid(a1*Theta1');
    a2 = [1 a2];
    a3 = sigmoid(a2*Theta2');
    [maximum , index] = max(a3);
    
    if ( index == 10)
        index = 0;
    end
    
    p(i) = index;

end
