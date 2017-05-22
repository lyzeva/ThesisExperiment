function Groundtruth = groundTruthByKNN(Test_data, Training_data, num_k)

euDistMat = distMat(Test_data,Training_data);
threshold = sort(euDistMat,2);
Groundtruth=(euDistMat<=repmat(threshold(:,num_k),[1,size(euDistMat,2)]));