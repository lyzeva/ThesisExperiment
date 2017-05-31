close all;
clc;
clear;

dataset = 'cifar-k';
addpath('./kerneldata/');
load([dataset,'.mat']);

hbits = 4; %[4, 8, 10, 14, 16];
num_test = 662;                % 1000 query test point, rest are database
intv = 100; PRline = 8457;     %PR������ʾ����

X = X';
ndata = size(X,2);
num_train = ndata - num_test;
%-- center the data
for i = 1:size(X,1)
    X(i,:) = X(i,:) - mean(X(i,:));
end


disp('Calculate groundtruth...');
Groundtruth =  groundTruthOnLabel(label(num_train+1:end),label(1:num_train));
%����С��1��Ϊ�棬ֵΪ1������Ϊ0

    bit = hbits*hbits;
    [Dhamm] = LDA2D(double(X),label,Groundtruth,n,hbits);
    [recall, precision, thresh_hball] = recall_precision(Groundtruth, Dhamm);
    [recall_kNN, precision_kNN, thresh_kNN] = recall_precision_kNN(Groundtruth, Dhamm,intv,PRline);
    map = area_RP(recall, precision);
    
    
    case 'Kernel-ITQ'
        addpath('./ITQ');
        tic;
        [pc, l] = eigs(cov(param.KTrain), bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
        Y1 = param.KTrain * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
        % ITQ
        [B, R] = ITQ(Y1, 50);   %50Ϊ������RΪ���50�κ�õ�����ת����
        B1 = compactbit(B);
        training_time = toc + param.anchor_traintime;
        P = pc*R;
        tic;
        Y2 = param.KTest*P;      %�������ɷֽ���ͶӰ���������ݵ��ٽ�����ת�任
        B2 = compactbit(Y2>0);
        coding_time = toc + param.anchor_testtime;
        % compute Hamming metric and compute recall precision
        memory = length(P(:))*8;
        Dhamm = hammingDist(B2, B1);
        
    case 'Kernel-LSH'
        tic;
        W = randn(param.num_anchor, bit);
        Y1 = (param.KTrain*W >= 0);    %ԭ������Ļ�����ֵ��Ϊ0������0����Ϊ1��С��0����Ϊ0
        B1 = compactbit(Y1);
        training_time = toc + param.anchor_traintime;
        tic;
        Y2 = (param.KTest*W >= 0);
        B2 = compactbit(Y2);
        coding_time = toc + param.anchor_testtime;
        Dhamm = hammingDist(B2, B1);
        memory = length(W(:))*8;
        
        