function Groundtruth = groundTruthByDistance(Test_data, Training_data)
averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor
% define ground-truth neighbors (this is only used for the evaluation):
num_training = size(Training_data,2);
R = randperm(num_training);  %����1��59000���������
DtrueTraining = distMat(Training_data(:,R(1:100)),Training_data); % sample 100 points to find a threshold
Threshold = sort(DtrueTraining,2); %DtrueTraining������С��������
clear DtrueTraining;
Threshold = mean(Threshold(:,averageNumberNeighbors)); %ȡ��50�в���ƽ��
% scale data so that the target distance is 1
Training_data = Training_data / Threshold;
Test_data = Test_data / Threshold;
Threshold = 1;

DtrueTestTraining = distMat(Test_data,Training_data);  %����Xtest��Training_data��ŷʽ����
Groundtruth = DtrueTestTraining < Threshold; %����С��1��Ϊ�棬ֵΪ1������Ϊ0




