load mnist.mat

r = 28;
m = 28;
num = 60000;


sampleMean = mean(X,2);
X = (X-repmat(sampleMean, 1, size(X,2)));

XX = reshape(X, m, r*num)';
for i=1:num
    X(:,i) = reshape(XX( (i-1)*r+1:i*r, :), m*r, 1);
end   

save mnistT.mat X label
