function [B,R] = ITQ(V, n_iter)
%
% main function for ITQ which finds a rotation of the PCA embedded data
% Input:
%       V: n*c PCA embedded data, n is the number of images and c is the
%       code length
%       n_iter: max number of iterations, 50 is usually enough
% Output:
%       B: n*c binary matrix
%       R: the c*c rotation matrix found by ITQ
% Author:
%       Yunchao Gong (yunchao@cs.unc.edu)
% Publications:
%       Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
%       Procrustes Approach to Learning Binary Codes. In CVPR 2011.
%

% initialize with a orthogonal random rotation
bit = size(V,2);
R = randn(bit,bit);
[U11 S2 V2] = svd(R);
R = U11(:,1:bit);
function_value = zeros(1,n_iter+1);
n_iter = 50;
% ITQ to find optimal rotation
for ITQiter=0:n_iter
    ITQiter
    Z = V * R;   %�������ɷֽ���ͶӰ���������ݵ��ٽ�����ת�任      
    UX = ones(size(Z,1),size(Z,2)).*-1;  
    UX(Z>=0) = 1;  %UX�е�Ԫ��Ϊ1��-1
    function_value(ITQiter+1) = sum(sum((UX-Z).^2));
    C = UX' * V;
    [UB,sigma,UA] = svd(C);    
    R = UA * UB';
end

% figure;
% plot(1:ITQiter+1,function_value);
% xlabel('ITQiter');


% make B binary
B = UX;
B(B<0) = 0;