function Groundtruth = groundTruthKNN(Test_data, Training_data, num_k)

euDistMat = distMat(Test_data,Training_data);
[~,perm] = sort(euDistMat,2);
Groundtruth=perm(:,1:num_k);