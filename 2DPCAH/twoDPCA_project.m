function Y = twoDPCA_project(A, r, W)%XÊÇÊú×ÅµÄ
% AA = reshape(A, m, n*num);
% [dim, num] = size(X);
% n = ceil(dim/m);
% XX = [X;zeros(m*n-dim,num)];
% A = reshape(X,m,n,num);
Y = A*W;
% Y = zeros(m, r, num);
% for i=1:num
%     Y(:,:,i) = A(:,:,i)'*W;
% end
% 