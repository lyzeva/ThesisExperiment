%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function: this is a geometric illustration of Draw the recall precision curve
%Author: Willard (Yuan Yong' English Name)
%Date: 2013-07-22
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clc;
clear;
addpath('./utils');
% --load data
% assess.X:dataset
%    label:class lable
%    ndata:number of data
%        D:dimension of data

%experiment arguments
assess.dataset = 'AwA'
load(['../Data/',assess.dataset,'.mat']);
assess.method = { 'BDAH^0', 'BDAH', 'CCA-ITQ', 'KSH', 'PCA-ITQ', 'LSH'};%  , 'PCAH',  'BDAH', 'KSH', 'CBE-opt', 'CCA-ITQ'
% assess.method = { 'LDA', 'CCA', 'Euclidean Distance'}; % '2DLDA', '2DLDA-LDA'
assess.num_methods = length(assess.method);
assess.hbits = [16 25 64 100];
assess.intv = 1; assess.PRline =500;     %interval and PRline
assess.loop = 1;

disp('Preprocess the data into normative form..');
param.num_test = 460;                % 1000 query test point, rest are database
param.num_train = 9000;

for loop = 1:assess.loop

    [param.X, param.num_train, param.label] = preprocess(X, param.num_test, param.num_train, label);
    clear X;
    % [param.X, param.num_train] = preprocess(X, param.num_test, param.num_train);

    disp('Calculate groundtruth...');
    assess.Groundtruth =  groundTruthOnLabel(param.label(param.num_train+1:end),param.label(1:param.num_train)); 
    % assess.NN=20;
    % param.Groundtruth =  groundTruthByKNN(param.X(:,param.num_train+1:end),param.X(:,1:param.num_train) ,assess.NN); 
    % assess.Groundtruth =  groundTruthByRandomDistance(param.X(:,param.num_train+1:end), param.X(:,1:param.num_train)); 

% for i=1:assess.length2D
%     for j = 1:length(assess.hbits)
%         disp(assess.method{i});
%         param.n = n;
%         param.bit = assess.hbits(j);
%         [res.Dhamm, res.training_time{j,i}, res.coding_time{j,i}, memory{j,i}] = hashCalculator2D( param, assess.method{i});
%         [recall{j,i}, precision{j,i}, thresh_hball{j,i}] = recall_precision(param.Groundtruth, res.Dhamm);
%         [recall_kNN{j,i}, precision_kNN{j,i}, thresh_kNN{j,i}] = recall_precision_kNN(param.Groundtruth, res.Dhamm, assess.intv, assess.PRline);
% %         map{j,i} = area_RP(recall{j,i}, precision{j,i});
%     end
% end
    
    for i=1:assess.num_methods
        for j = 1:length(assess.hbits)
            disp(assess.method{i});
            param.bits = assess.hbits(j);
            param.r = sqrt(param.bits);
            param.n = n;
            [res.Dhamm, res.training_time{loop}{j,i},res.coding_time{loop}{j,i}] = hashCalculator(param, assess.method{i});
            [res.recall{loop}{j,i}, res.precision{loop}{j,i}, res.thresh_hball{loop}{j,i}] = recall_precision(assess.Groundtruth, res.Dhamm);
%             [res.recall_kNN{loop}{j,i}, res.precision_kNN{loop}{j,i}, res.thresh_kNN{loop}{j,i}] = recall_precision_kNN(assess.Groundtruth, res.Dhamm, assess.intv, assess.PRline);
            res.map{loop}{j,i} = area_RP(res.recall{loop}{j,i}, res.precision{loop}{j,i});
       end
    end
end

save([assess.dataset,'_res.mat'],'assess', 'res', 'param');
% 
% num_anchor = 1024;
% param.n = 32;
% 
% disp('preprocess of anchors..');
% tic;
% sample = randperm(param.num_train, num_anchor);
% anchor = param.X(:,sample);
% KTrain = sqdist(param.X(:,1:param.num_train),anchor);
% sigma = mean(mean(KTrain,2));
% KTrain = exp(-KTrain/(2*sigma));
% mvec = mean(KTrain);
% KTrain = (KTrain-repmat(mvec,param.num_train,1))';
% assess.anchor_traintime = toc;
% tic;
% KTest = sqdist(param.X(:,param.num_train+1:end),anchor);
% KTest = exp(-KTest/(2*sigma));
% KTest = (KTest - repmat(mvec,param.num_test,1))';
% assess.anchor_testtime = toc;
% param.X = [KTrain,KTest];
% clear KTrain;
% clear KTest;
% 
% lm = size(assess.method, 2);
% for i=1:size(assess.method,2)
%     assess.method{i+lm} = ['Kernel-',assess.method{i}];
% end



% save([dataset,'result.mat'],'dataset', 'assess', 'recall','precision','thresh_hball', 'map', 'recall_kNN','precision_kNN','thresh_kNN','res.training_time','res.coding_time','memory');
% 

% %memory vs. MAP
% figure; hold on;  
% for i = 1:size(method,2)
% 	time = zeros(1,length(hbits));
%     MAP = [];
% 	for j= 1:length(hbits)
% 		time(j) = memory{j,i};
%         MAP = [MAP, map{j, i}];
% 	end
% 	plot(MAP,time,[color(i),'-o'],'linewidth',1);
% end
% 	xlabel('MAP');
% 	ylabel('Memory Consumption��Byte)');
% 	title(dataset);
% 	legend(method);
% 	box on; hold off;
% 
