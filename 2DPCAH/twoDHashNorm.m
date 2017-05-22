function [recall, precision] = twoDHashNorm(XX, a, b, Groundtruth, r)

[num_test, num_training] = size(Groundtruth);
A = reshape(XX,a,b,num_test+num_training);

       [Y1, W] = twoDPCA(A(:,:,1:num_training), r);
       Y1 = reshape(sqrt(sum(reshape(Y1.*Y1, a, num_training*r),1)), num_training, r);
       [Y1, Mean] = center(Y1);
       [B1, R] = ITQ(Y1, 50);
       B1 = compactbit(B1);
       Y2 = twoDPCA_project(A(:,:,num_training+1:end), r, W);
       Y2 = reshape(sqrt(sum(reshape(Y2.*Y2, a, num_test*r),1)), num_test, r);
       Y2 = center(Y2);
       B2 = compactbit(Y2*R>0);
       
       


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(Groundtruth, Dhamm);