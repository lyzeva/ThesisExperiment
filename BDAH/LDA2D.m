function [Dhamm] = LDA2D(XX, label, Groundtruth, n, r)
[num_test, num_train] = size(Groundtruth);
[dim,num] = size(XX);

m = ceil(dim/n);
X = [XX;zeros(n*m-dim,num)];
clear XX;

X_train = X(:,1:num_train);
X_test = X(:,num_train+1:end);

[R, L] = iterative2DLDA( X_train, label(1:num_train),r,r,n,m);
        Y1 = zeros(r*r, num_train);
        for i=1:num_train
            Y1(:,i) = reshape(L'*reshape(X_train(:,i),n,m)*R, r*r, 1);
        end
        W = LDA(Y1, label(1:num_train), r*r);
        YY1 = Y1'*W;
        [B, U] = ITQ(YY1, 50);
        B1 =  compactbit(B);
        P = W*U;
        clear W;        clear U;        clear Y1;        clear YY1;
        tic;
        Y2 = zeros(r*r, num_test);
        for i=1:num_test
            Y2(:,i) = reshape(L'*reshape(X_test(:,i),n,m)*R, r*r, 1);
        end
        B2 =  compactbit(Y2'*P>0);
        
        Dhamm = hammingDist(B2, B1);
 