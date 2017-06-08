function [recall, precision, thresh] = recall_precision(Wtrue, Dhat, intv)
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

if (nargin < 3)
    intv = 1;
end
disp('recall precsion..');
max_hamm = max(Dhat(:)); %��Dhat(:)��������һ��1ά��������Ȼ��ȡ���ֵ
hamm_thresh = min(3,max_hamm);%�Ƚ�3,max_hamm��ȡ���ߵ���Сֵ

[Ntest, Ntrain] = size(Wtrue);
total_good_pairs = sum(Wtrue(:)); %��Wtrue(:)��������һ��1ά��������Ȼ�����
%ԭŷʽ�ռ���С��Dball�ĸ���WtrueTestTraining = DtrueTestTraining < Dball

% find pairs with similar codes
thresh = 1:intv:max_hamm;
precision = zeros(length(thresh),1);
recall = zeros(length(thresh),1);

for n = 0:length(thresh)
    j = (Dhat<=((thresh(n)-1)+0.00001));
    
    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = sum(Wtrue(j)); %�ҳ�Wtrue����jΪ�������
    
    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = sum(j(:)); %ͳ��j��1�ĸ���

    precision(n) = retrieved_good_pairs/retrieved_pairs;
    recall(n)= retrieved_good_pairs/total_good_pairs;
end

% The standard measures for IR are recall and precision. Assuming that:
%
%    * RET is the set of all items the system has retrieved for a specific inquiry;
%    * REL is the set of relevant items for a specific inquiry;
%    * RETREL is the set of the retrieved relevant items 
%
% then precision and recall measures are obtained as follows:
%
%    precision = RETREL / RET
%    recall = RETREL / REL 

% if nargout == 0 || nargin > 3
%     if isempty(fig);
%         fig = figure;
%     end
%     figure(fig)
%     
%     subplot(311)
%     plot(0:hamm_thresh-1, precision(1:hamm_thresh), varargin{:})
%     hold on
%     xlabel('hamming radius')
%     ylabel('precision')
%     
%     subplot(312)
%     plot(0:hamm_thresh-1, recall(1:hamm_thresh), varargin{:})
%     hold on
%     xlabel('hamming radius');
%     ylabel('recall');
%         
%    subplot(313);
%     plot(recall, precision, varargin{:});
%     hold on;
%     axis([0 1 0 1]);
%     xlabel('recall');
%     ylabel('precision');
% 
%     drawnow;
% end
