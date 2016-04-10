function g = sigmoidGradient(z)
g=sigmoid(z);
g=g.*(1-g);
end
