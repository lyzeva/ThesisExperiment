function [Dhamm, training_time, coding_time, memory] = hashCalculator2D(param, method)%XX, label, Groundtruth, n, r, 
[dim,num] = size(param.X);
[num_test, num_train] = size(param.Groundtruth);

n = param.n;
m = ceil(dim/n);
X = [param.X;zeros(n*m-dim,num)];
X_train = X(:,1:num_train);
X_test = X(:,num_train+1:end);
clear X;

r = sqrt(param.bit);

switch(method)

    case 'BDAH'
        %--center the data
        tic;
        [R, L] = iterative2DLDA(X_train, label(1:num_train),r,r,n,m);
        Y1 = zeros(r*r, num_train);
        for i=1:num_train
            Y1(:,i) = reshape(L'*reshape(X_train(:,i),n,m)*R, r*r, 1);
        end
        [B, U] = ITQ(Y1', 50);
        B1 =  compactbit(B);
        training_time = toc

        tic;
        Y2 = zeros(r*r, num_test);
        for i=1:num_test
            Y2(:,i) = reshape(L'*reshape(X_test(:,i),n,m)*R, r*r, 1);
        end
        B2 = compactbit(Y2'*U>0);
        coding_time = toc
        memory = (length(L(:))+length(R(:))+length(U(:)))*8;
        Dhamm = hammingDist(B2, B1);
    case 'BDAH^0'
        tic;
        r = 2*r;
        %--center the data
        [R, L] = iterative2DLDA( X_train, label(1:num_train),r,r,n,m);
        Y1 = zeros(r*r, num_train);
        for i=1:num_train
            Y1(:,i) = reshape(L'*reshape(X_train(:,i),n,m)*R, r*r, 1);
        end
        r = r/2;
        W = LDA(Y1, label(1:num_train), r*r);
        YY1 = Y1'*W;
        [B, U] = ITQ(YY1, 50);
        B1 =  compactbit(B);
        training_time = toc
        P = W*U;
        clear W;        clear U;        clear Y1;        clear YY1;
        tic;
        r=r*2;
        Y2 = zeros(r*r, num_test);
        for i=1:num_test
            Y2(:,i) = reshape(L'*reshape(X_test(:,i),n,m)*R, r*r, 1);
        end
        r = r/2;
        B2 =  compactbit(Y2'*P>0);
        
        coding_time = toc
        memory = (length(L(:))+length(R(:))+length(P(:)))*8;
        Dhamm = hammingDist(B2, B1);
    case '2DPCA-2DITQ'
        tic;
        %--1D to connected 2D
        A = reshape(X,n,m*num)';
       [Y1, W] = twoDPCA(A(1:num_train*m,:), r, num_train);
       [B1, R] = ITQ(Y1, 50);
       B1 = compactbit(reshape(B1',m*r,num_train)');
       training_time = toc

       tic;
       Y2 = twoDPCA_project(A(num_train*m+1:end,:), r, W);
       B2 = compactbit(reshape((Y2*R>0)',m*r,num_test)');
       coding_time = toc
       Dhamm = hammingDist(B2, B1);
    case '2DLDAH'
        tic;
        %--center the data
        [R, L] = iterative2DLDA(X(:,1:num_train),label(1:num_train),r,r,n,m);
        Y1 = zeros(r*r, num_train);
        Y2 = zeros(r*r, num_test);
        for i=1:num_train
            Y1(:,i) = reshape(L'*reshape(X(:,i),n,m)*R, r*r, 1);
        end
        B1 = compactbit(Y1'>0);
        training_time = toc

        tic;
        for i=1:num_test
            Y2(:,i) = reshape(L'*reshape(X(:,i+num_train),n,m)*R, r*r, 1);
        end
        B2 = compactbit(Y2'>0);
        coding_time = toc
        Dhamm = hammingDist(B2, B1);
    case '2DLDA-LDAH'
        tic;
        r = 2*r;
        [R, L] = iterative2DLDA(X(:,1:num_train),label(1:num_train),r,r,n,m);
        Y = zeros(r*r, num);
        for i=1:num
            Y(:,i) = reshape(L'*reshape(X(:,i),n,m)*R, r*r, 1);
        end
        r = r/2;
        W = LDA(Y, label, r*r);
        YY = W'*Y;
        B1 = YY(:,1:num_train)>0;
        B1 = compactbit(B1');
        training_time = toc
    
        tic;
        B2 = YY(:,num_train+1:end)>0;
        B2 = compactbit(B2');
        coding_time = toc
        Dhamm = distMat(YY(:,num_train+1:end),YY(:,1:num_train));  
    case '2DPCA'
        %--1D to connected 2D
        tic;
        r = param.bit/m;
        A1 = reshape(X_train,n,m*num_train)';
        [Y1, W] = twoDPCA(A1, r, num_train);
        B1 = reshape(Y1'>0,m*r,num_train);
        B1 = compactbit(B1');
        training_time = toc;
        tic;
        A2 = reshape(X_test,n,m*num_test)';
        Y2 = twoDPCA_project(A2, r, W);
        B2 = reshape(Y2'>0,m*r,num_test);
        B2 = compactbit(B2');
        coding_time = toc;
        Dhamm = hammingDist(B2, B1);
        memory = 0;
    case '2DLDA'
        tic;
        [R, L] = iterative2DLDA(X(:,1:num_train),label(1:num_train),r,r,n,m);
        Y1 = zeros(r*r, num_train);
        Y2 = zeros(r*r, num_test);
        for i=1:num_train
            Y1(:,i) = reshape(L'*reshape(X(:,i),n,m)*R, r*r, 1);
        end
        training_time = toc

        for i=1:num_test
            Y2(:,i) = reshape(L'*reshape(X(:,i+num_train),n,m)*R, r*r, 1);
        end
        Dhamm = distMat(Y2, Y1);
    case '2DLDA-LDA'
        tic;
        r = 2*r;
        [R, L] = iterative2DLDA(X(:,1:num_train),label(1:num_train),r,r,n,m);
        Y = zeros(r*r, num);
        for i=1:num
            Y(:,i) = reshape(L'*reshape(X(:,i),n,m)*R, r*r, 1);
        end
        r = r/2;
        W = LDA(Y, label, r*r);
        YY = W'*Y;
        training_time = toc

        Dhamm = distMat(YY(:,num_train+1:end),YY(:,1:num_train));  
    case 'None'
        training_time = 0;
        coding_time = 0;
        Dhamm = distMat(X(:,num_train+1:end)',X(:,1:num_train)');
end
