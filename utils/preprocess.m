function [Xout, num_train, Lout] =  preprocess(X, num_test, num_train, label)

ndata = size(X,2);
if ndata < num_train + num_test
    num_train = ndata - num_test;
end
% if num_train>10000
%     num_train = 10000;
% end

%-- center the data
for i = 1:size(X,1)
    X(i,:) = X(i,:) - mean(X(i,:));
end

%-- split up into training arend test set
R = randperm(ndata);
%R = 1:ndata;
Xout = X(:,R(1:num_train+num_test));
if nargin > 3
    Lout = label(R(1:num_train+num_test));
end
