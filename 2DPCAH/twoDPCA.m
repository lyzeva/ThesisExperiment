function [Y, W]=twoDPCA(A, r, num)%A是图片，在第一维排列； W是映射矩阵
G = A'*A;
% for i=1:num
%     G = G + A(:,:,i)'*A(:,:,i);
% end
Gt = G/num;
[W D] = eigs(Gt,r);
Y = A*W;
% Y = zeros(m, r, num);
% for i=1:num
%      Y(:,:,i) = A(:,:,i)'*W;
% end
