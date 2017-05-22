close all;
clc;
clear;

dataset = 'cifar-k';
addpath('./kerneldata/');
load([dataset,'.mat']);

hbits = 4; %[4, 8, 10, 14, 16];
num_test = 662;                % 1000 query test point, rest are database
intv = 100; PRline = 8457;     %PR曲线显示长度

X = X';
ndata = size(X,2);
num_train = ndata - num_test;
%-- center the data
for i = 1:size(X,1)
    X(i,:) = X(i,:) - mean(X(i,:));
end


disp('Calculate groundtruth...');
Groundtruth =  groundTruthOnLabel(label(num_train+1:end),label(1:num_train));
%满足小于1的为真，值为1；否则为0

    bit = hbits*hbits;
    [Dhamm] = LDA2D(double(X),label,Groundtruth,n,hbits);
    [recall, precision, thresh_hball] = recall_precision(Groundtruth, Dhamm);
    [recall_kNN, precision_kNN, thresh_kNN] = recall_precision_kNN(Groundtruth, Dhamm,intv,PRline);
    map = area_RP(recall, precision);