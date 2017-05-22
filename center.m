function [Ac, mean] = center(A)
n = size(A,1);
mean = sum(A)/n;
Ac = A - repmat(mean, n, 1);
