function [Dhamm] = twoDHashITQ(XX, Groundtruth, n, r)

[num_test, num_training] = size(Groundtruth);
num = num_test+ num_training;
[dim,~] = size(XX);
m = ceil(dim/n);
X = [XX;zeros(m*n-dim,num)];
A = reshape(X,n,m*num)';
clear X;
clear XX;
       [Y1, W] = twoDPCA(A(1:num_training*m,:), r, num_training);
       YY1 = zeros(m*r,num_training);
       for i=1:num_training
           YY1(:,i) = reshape(Y1( (i-1)*m+1:i*m, :), m*r, 1);
       end
       Y1 = YY1';
       [B1, R] = ITQ(Y1, 50);
       B1 = compactbit(B1);
       Y2 = twoDPCA_project(A(num_training*m+1:end,:), r, W);
       YY2 = zeros(m*r,num_test);
       for i=1:num_test
           YY2(:,i) = reshape(Y2( (i-1)*m+1:i*m, :), m*r, 1);
       end
       Y2 = YY2';
       B2 = compactbit(Y2*R>0);


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);