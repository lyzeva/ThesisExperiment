function Groundtruth = groundTruthByDistance(Test_data, Training_data)
averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor
% define ground-truth neighbors (this is only used for the evaluation):
num_training = size(Training_data,2);
R = randperm(num_training);  %生成1到59000的随机序列
DtrueTraining = distMat(Training_data(:,R(1:100)),Training_data); % sample 100 points to find a threshold
Threshold = sort(DtrueTraining,2); %DtrueTraining按行由小到大排序
clear DtrueTraining;
Threshold = mean(Threshold(:,averageNumberNeighbors)); %取第50列并求平均
% scale data so that the target distance is 1
Training_data = Training_data / Threshold;
Test_data = Test_data / Threshold;
Threshold = 1;

DtrueTestTraining = distMat(Test_data,Training_data);  %计算Xtest和Training_data的欧式距离
Groundtruth = DtrueTestTraining < Threshold; %满足小于1的为真，值为1；否则为0




