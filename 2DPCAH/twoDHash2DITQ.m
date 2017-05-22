function [Dhamm] = twoDHash2DITQ(XX, Groundtruth, n, r)

[num_test, num_training] = size(Groundtruth);
num = num_test+ num_training;
[dim,~] = size(XX);

m = ceil(dim/n);
X = [XX;zeros(n*m-dim,num)];
A = reshape(X,n,m*num)';
clear X;

       [Y1, W] = twoDPCA(A(1:num_training*m,:), r, num_training);
%        Y1 = reshape(Y1,m*r,num_training);
    
       [B1, R] = ITQ(Y1, 50);
       B1 = compactbit(reshape(B1',m*r,num_training)');
       Y2 = twoDPCA_project(A(num_training*m+1:end,:), r, W);
       B2 = compactbit(reshape((Y2*R>0)',m*r,num_test)');
       
       


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);