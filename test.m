function [recall, precision] = test(Database, bit, method)
%
% demo code for generating small code and evaluation
% input X should be a n*d matrix, n is the number of images, d is dimension
% ''method'' is the method used to generate small code
% ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH' 

% parameters
averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor
num_test = 1000;                % 1000 query test point, rest are database


% split up into training and test set
load(Database);

[ndata, D] = size(X);
R = randperm(ndata); %1��n��Щ��������ҵõ���һ���������������Ϊ����
Xtest = X(R(1:num_test),:);  %��������ǰ1000�����ݵ���Ϊ��������
Ltest = label(R(1:num_test));
R(1:num_test) = [];
Xtraining = X(R,:);   %ʣ�µ�������Ϊѵ������
Ltraining = label(R,:);
num_training = size(Xtraining,1);
clear X;
clear R;


% threshold to define ground truth
% WtrueTestTraining =  groundTruthByDistance(Xtest,Xtraining,Dball); %����С��1��Ϊ�棬ֵΪ1������Ϊ0
WtrueTestTraining =  groundTruthOnLabel(Ltest,Ltraining,); %����С��1��Ϊ�棬ֵΪ1������Ϊ0

% generate training ans test split and the data matrix
XX = [Xtraining; Xtest];
% center the data, VERY IMPORTANT
sampleMean = mean(XX,1);
XX = (double(XX)-repmat(sampleMean,size(XX,1),1));


%several state of art methods
switch(method)
    
    % ITQ method proposed in our CVPR11 paper
    case 'ITQ'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);%��database�е����ݵ����Э�����Э������������ֽ��õ����ɷ�
        XX = XX * pc;  %�������ݵ������ɷ��Ͻ���ͶӰ
        % ITQ
        [Y, R] = ITQ(XX(1:num_training,:),50);   %50Ϊ����������RΪ����50�κ�õ�����ת����
        XX = XX*R;      %�������ɷֽ���ͶӰ����������ݵ��ٽ�����ת�任
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);
        % Our method 
    case 'QQ'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);  %��database�е����ݵ����Э�����Э������������ֽ��õ����ɷ�
        XX = XX * pc;  %�������ݵ������ɷ��Ͻ���ͶӰ
        % QQ
        [Y, R] = QQ(XX(1:num_training,:));   %50Ϊ����������RΪ����50�κ�õ�����ת����
        XX = XX*R;      %�������ɷֽ���ͶӰ����������ݵ��ٽ�����ת�任
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);
    % RR method proposed in our CVPR11 paper
    case 'RR'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)), bit);
        XX = XX * pc;
        % RR
        R = randn(size(XX,2),bit);
        [U S V] = svd(R);
        XX = XX*U(:,1:bit);
        Y = compactbit(XX>0);
   % SKLSH
   % M. Raginsky, S. Lazebnik. Locality Sensitive Binary Codes from
   % Shift-Invariant Kernels. NIPS 2009.
    case 'SKLSH' 
        RFparam.gamma = 1;
        RFparam.D = D;
        RFparam.M = bit;
        RFparam = RF_train(RFparam);
        B1 = RF_compress(XX(1:num_training,:), RFparam);
        B2 = RF_compress(XX(num_training+1:end,:), RFparam);
        Y = [B1;B2];
    % Locality sensitive hashing (LSH)
     case 'LSH'
        XX = XX * randn(size(XX,2),bit);%randn(size(XX,2),bit)����w,wΪ320*24
        Y = zeros(size(XX));
        Y(XX>=0)=1;    %ԭ�������Ļ�����ֵ��Ϊ0������0����Ϊ1��С��0����Ϊ0
        Y = compactbit(Y);
     case  'PCAH'
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);%��database�е����ݵ����Э�����Э������������ֽ��õ����ɷ�
        XX = XX * pc;  %�������ݵ������ɷ��Ͻ���ͶӰ
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);       
end

% compute Hamming metric and compute recall precision
B1 = Y(1:size(Xtraining,1),:);        %������ѵ������
B2 = Y(size(Xtraining,1)+1:end,:);    %�����Ĳ�������
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(WtrueTestTraining, Dhamm);