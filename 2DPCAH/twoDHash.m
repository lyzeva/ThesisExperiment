function [Dhamm] = twoDHash(XX, Groundtruth, n, r)
%
% demo code for generating small code and evaluation
% input X should be a n*d matrix, n is the number of images, d is dimension
% ''method'' is the method used to generate small code
% ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH' 

% % parameters
% averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor
% num_test = 1000;                % 1000 query test point, rest are database
% 
% 
% % split up into training and test set
% [ndata, D] = size(X);
% R = randperm(ndata); %1��n��Щ��������ҵõ���һ���������������Ϊ����
% Xtest = X(R(1:num_test),:);  %��������ǰ1000�����ݵ���Ϊ��������
% R(1:num_test) = [];
% Xtraining = X(R,:);   %ʣ�µ�������Ϊѵ������
% num_training = size(Xtraining,1);
% clear X;
% 

% 
% % generate training ans test split and the data matrix
% XX = [Xtraining; Xtest];
% % center the data, VERY IMPORTANT
% sampleMean = mean(XX,1);
% XX = (double(XX)-repmat(sampleMean,size(XX,1),1));
% ����Ԥ����
[num_test, num_training] = size(Groundtruth);
num = num_test+num_training;
[dim,~] = size(XX);
m = ceil(dim/n);
X = [XX;zeros(m*n-dim, num)];
A = reshape(X,n,num*m)';
% PCA%ITQ
       [Y1, W] = twoDPCA(A(1:num_training*m,:), r, num_training);
       B1 = compactbit(reshape((Y1>0)',m*r,num_training)');
       Y2 = twoDPCA_project(A(num_training*m+1:end,:), r, W);
       B2 = compactbit(reshape((Y2>0)',m*r,num_test)');
       
       


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);