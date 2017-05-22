load cifar10data.mat

r = 32;
m = 32;
num = 50004;

sampleMean = mean(X,2);
X = (X-repmat(sampleMean, 1, size(X,2)));

R = randperm(num); %1��n��Щ��������ҵõ���һ���������������Ϊ����
X = X(:,R);
label = label(R);

save cifar10_c.mat X label;

XX = reshape(X, m, r*num)';
for i=1:num
    X(:,i) = reshape(XX( (i-1)*r+1:i*r, :), m*r, 1);
end   

save cifar10T_c.mat X label;
