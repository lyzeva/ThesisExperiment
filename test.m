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
R = randperm(ndata); %1到n这些数随机打乱得到的一个随机数字序列作为索引
Xtest = X(R(1:num_test),:);  %以索引的前1000个数据点作为测试样本
Ltest = label(R(1:num_test));
R(1:num_test) = [];
Xtraining = X(R,:);   %剩下的数据作为训练样本
Ltraining = label(R,:);
num_training = size(Xtraining,1);
clear X;
clear R;


% threshold to define ground truth
% WtrueTestTraining =  groundTruthByDistance(Xtest,Xtraining,Dball); %满足小于1的为真，值为1；否则为0
WtrueTestTraining =  groundTruthOnLabel(Ltest,Ltraining,); %满足小于1的为真，值为1；否则为0

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
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);%对database中的数据点计算协方差，对协方差进行特征分解后得到主成分
        XX = XX * pc;  %所有数据点在主成分上进行投影
        % ITQ
        [Y, R] = ITQ(XX(1:num_training,:),50);   %50为迭代次数，R为迭代50次后得到的旋转矩阵
        XX = XX*R;      %对在主成分进行投影后的所有数据点再进行旋转变换
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);
        % Our method 
    case 'QQ'
        % PCA
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);  %对database中的数据点计算协方差，对协方差进行特征分解后得到主成分
        XX = XX * pc;  %所有数据点在主成分上进行投影
        % QQ
        [Y, R] = QQ(XX(1:num_training,:));   %50为迭代次数，R为迭代50次后得到的旋转矩阵
        XX = XX*R;      %对在主成分进行投影后的所有数据点再进行旋转变换
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
        XX = XX * randn(size(XX,2),bit);%randn(size(XX,2),bit)生成w,w为320*24
        Y = zeros(size(XX));
        Y(XX>=0)=1;    %原数据中心化后，阈值设为0。大于0编码为1，小于0编码为0
        Y = compactbit(Y);
     case  'PCAH'
        [pc, l] = eigs(cov(XX(1:num_training,:)),bit);%对database中的数据点计算协方差，对协方差进行特征分解后得到主成分
        XX = XX * pc;  %所有数据点在主成分上进行投影
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);       
end

% compute Hamming metric and compute recall precision
B1 = Y(1:size(Xtraining,1),:);        %编码后的训练样本
B2 = Y(size(Xtraining,1)+1:end,:);    %编码后的测试样本
Dhamm = hammingDist(B2, B1);
[recall, precision, rate] = recall_precision(WtrueTestTraining, Dhamm);