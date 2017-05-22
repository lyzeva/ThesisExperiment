function [recall, precision] = twoDBITQ(XX, Groundtruth, m, b1, b2)


[num_test, num_training] = size(Groundtruth);
[dim,num] = size(XX);

n = ceil(dim/m);
X = [XX;zeros(n*m-dim,num)];
A = reshape(X,m,n,num);
clear X;

       [R1, R2] =  BilinearITQ_low(A(:,:,1:num_training), b1, b2, 50);
%        Y1 = reshape(Y1,m*r,num_training);
       B = logical(zeros(num_training, b1*b2));
       for i=1:num
           B(i,:) = reshape(R1'*A(:,:,i)*R2>0, 1, b1*b2);
       end
       B1 =  compactbit(B(1:num_training,:));
       B2 =  compactbit(B(num_training+1:end,:));


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(Groundtruth, Dhamm,3);