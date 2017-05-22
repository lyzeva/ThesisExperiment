function Dist = twoDHashPQ(XX, Groundtruth, n, r, bit)
[num_test, num_training] = size(Groundtruth);
num = num_test+ num_training;
[dim,~] = size(XX);
m = ceil(dim/n);
X = [XX;zeros(m*n-dim,num)];
A = reshape(X,n,m*num)';
clear X;
clear XX;
       [Y1, W] = twoDPCA(A(1:num_training*m,:), r, num_training);
       YY1 = zeros(m*r,num_training);
       for i=1:num_training
           YY1(:,i) = reshape(Y1( (i-1)*m+1:i*m, :), m*r, 1);
       end
       [centers_table, B1] = PQtrain( YY1, r, 2.^bit/r);
       Y2 = twoDPCA_project(A(num_training*m+1:end,:), r, W);
       YY2 = zeros(m*r,num_test);
       for i=1:num_test
           YY2(:,i) = reshape(Y2( (i-1)*m+1:i*m, :), m*r, 1);
       end
       PQCodesI=PQcoding( centers_table, YY2, 2.^bit/r);
%  Dist = zeros(size(Groundtruth));
%  for i=1:num_test
%      Dhamm =(sum(B1-repmat(PQCodesI(:,i),1,num_training),1)==0);
%      Dist(i,:) = sqrt(sum( ((YY1-repmat(YY2(:,i),1,num_training)).*repmat(Dhamm,m*r,1)).^2,1));
%  end
% Dist(find(Dist(:)==0)) = Inf;
     Dhamm = PQDist(PQCodesI, B1, centers_table);
 
%  
% 
% 
% % compute Hamming metric and compute recall precision
% Dhamm = hammingDist(B2, B1);
% [recall, precision, thresh] = recall_precision(Groundtruth, Dhamm, 1);
%         PQCodesT=PQcoding( centers_table, (dataT*PT)', 2^PQ_WORD );