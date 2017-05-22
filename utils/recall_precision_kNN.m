function [recall, precision, thresh] = recall_precision_kNN(Wtrue, Dhat, intv, PointsNum)
%
% Input:
%    Wtrue = true neighbors [Ntest * Ndataset], can be a full matrix NxN
%    Dhat  = estimated distances
%
% Output:
%
%                  exp. # of good pairs inside hamming ball of radius <= (n-1)
%  precision(n) = --------------------------------------------------------------
%                  exp. # of total pairs inside hamming ball of radius <= (n-1)
%
%               exp. # of good pairs inside hamming ball of radius <= (n-1)
%  recall(n) = --------------------------------------------------------------
%                          exp. # of total good pairs 
disp('recall precision kNN');
[Ntest, Ntrain] = size(Wtrue);

if (nargin<4)
    PointsNum = Ntrain;
end

if (nargin<3)
    intv = 1;
end

[D,c]= sort(Dhat, 2);
thresh = 1:intv:PointsNum;
precision = zeros(length(thresh),1);
recall = zeros(length(thresh),1);

total_good_pairs = sum(Wtrue(:)); %将Wtrue(:)按列拉成一个1维列向量，然后求和

for n = 1:length(thresh)
    j = (Dhat<=repmat(D(:,thresh(n)),[1,Ntrain]));
    
    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = sum(sum(Wtrue(j))); %找出Wtrue中以j为真的索引
    
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = sum(j(:)); %统计j中1的个数

    precision(n) = retrieved_good_pairs/retrieved_pairs;
    recall(n)= retrieved_good_pairs/total_good_pairs;
end

% for i=1:length(thresh)
%     g = thresh(i);
%     retrieved_good_pairs = sum(sum(D(:,1:g)));
%     [row, col] = size(D(:,1:g));
%     total_pairs = row*col;
%     recall(i) = retrieved_good_pairs/total_good_pairs;
%     precision(i) = retrieved_good_pairs/total_pairs;
%     rate(i) = total_good_pairs / (Ntest*Ntrain);
% end



% max_hamm = max(Dhat(:)); %将Dhat(:)按列拉成一个1维列向量，然后取最大值
% hamm_thresh = min(3,max_hamm);%比较3,max_hamm，取两者的最小值
% 
% [Ntest, Ntrain] = size(Wtrue);
% total_good_pairs = sum(Wtrue(:)); %将Wtrue(:)按列拉成一个1维列向量，然后求和
% %原欧式空间中小于Dball的个数WtrueTestTraining = DtrueTestTraining < Dball
% 
% % find pairs with similar codes
% 
% for n = 1:length(thresh)
%     j = (Dhat<=((thresh(n)-1)+0.00001));
%     
%     %exp. # of good pairs that have exactly the same code
%     retrieved_good_pairs = sum(Wtrue(j)); %找出Wtrue中以j为真的索引
%     
%     % exp. # of total pairs that have exactly the same code
%     retrieved_pairs = sum(j(:)); %统计j中1的个数
% 
%     precision(n) = retrieved_good_pairs/retrieved_pairs;
%     recall(n)= retrieved_good_pairs/total_good_pairs;
%     rate(n) = retrieved_pairs / (Ntest*Ntrain);
% end
