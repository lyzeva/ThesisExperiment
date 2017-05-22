       B1 = reshape(B1,m*r,num_training);
function [recall, precision] = twoDHash2(XX, a, b, Groundtruth, m, r)
%
% demo code for generating small code and evaluation
% input X should be a n*d matrix, n is the number of images, d is dimension
% ''method'' is the method used to generate small code
% ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH' 
% 
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


% % generate training ans test split and the data matrix
% XX = [Xtraining; Xtest];
% % center the data, VERY IMPORTANT
% sampleMean = mean(XX,1);
% XX = (double(XX)-repmat(sampleMean,size(XX,1),1));
[num_test, num_training] = size(Groundtruth);
Img = reshape(XX',a,b,num_test+num_training);
A = im2double(imresize(im2uint8(Img), [m,NaN]));

       [Y1, W] = twoDPCA(A(:,:,1:num_training), r);
       Y1 = reshape(Y1,m*r,num_training);
       [B1, R] = ITQ(Y1', 50);
       B1 = compactbit(B1);
       Y2 = twoDPCA_project(A(:,:,num_training+1:end), r, W);
       Y2 = reshape(Y2,m*r,num_test);
       B2 = compactbit(Y2'*R>0);
       
       


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(Groundtruth, Dhamm);