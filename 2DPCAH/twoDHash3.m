function [recall, precision] = twoDHash3(XX, a, b, Groundtruth, m, r)
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
       B1 = [];
       Mean = zeros(r,m);
       R = zeros(m,m,r);
       for i=1:r
            YY = reshape(Y1(:,i,:),m,num_training)'
            Mean(i,:) = mean(YY,1);
            YY = YY - repmat(Mean(i,:), num_training, 1);
            [BB, R(:,:,i)] = ITQ(YY, 50);
            B1 = [B1,BB];
       end
       B1 = compactbit(B1);
       Y2 = twoDPCA_project(A(:,:,num_training+1:end), r, W) - repmat(Mean',[1,1,num_test]);
       Y3 = [];
       for i = 1:r
           YY = reshape(Y2(:,i,:),m,num_test)'*R(:,:,i);
           Y3 = [Y3,YY];
       end
       B2 = compactbit(Y3>0);
       
       


% compute Hamming metric and compute recall precision
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(Groundtruth, Dhamm);