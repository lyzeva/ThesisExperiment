function [Dhamm, training_time, coding_time, memory] = hashCalculator(param, method)
% param.X, param.label, param.Groundtruth, bit, num_test
% demo code for generating small code and evaluation
% input X should be a n*d matrix, n is the number of images, d is dimension
% ''method'' is the method used to generate small code
% ''method'' can be 'ITQ', 'RR', 'LSH' and 'SKLSH' 
num_train = param.num_train;
num_test = param.num_test;
[dim, num] = size(param.X);
bit = param.bits;

%several state of art methods
switch(method)
    
     case 'LSH'
        X_train = (param.X(:,1:num_train));
        X_test = (param.X(:,num_train+1:end));
        clear param;

        tic;
        W = randn(bit, dim);
        Y1 = (W*X_train>=0)';    %ԭ������Ļ�����ֵ��Ϊ0������0����Ϊ1��С��0����Ϊ0
        B1 = compactbit(Y1);
        training_time = toc
        
        tic;
        Y2 = (W*X_test>=0)';
        B2 = compactbit(Y2);
        coding_time = toc

        Dhamm = hammingDist(B2, B1);
        memory = length(W(:))*8;
        
        
    % ITQ method proposed in our CVPR11 paper
    case 'PCA-ITQ'
        addpath('./ITQ');
        X_train = (param.X(:,1:num_train))';
        X_test = (param.X(:,num_train+1:end))';
        clear param;
        
        % training
        tic;
        [pc, l] = eigs(cov(X_train), bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
        Y1 = X_train * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
        [B, R] = ITQ(Y1, 50);   %50Ϊ������RΪ���50�κ�õ�����ת����
        B1 = compactbit(B);
        P = pc*R; %calculating projection matrix
        training_time = toc        
        
        % indexing test data
        tic;
        Y2 = X_test*P;      %�������ɷֽ���ͶӰ���������ݵ��ٽ�����ת�任
        B2 = compactbit(Y2>0);
        coding_time = toc
        
        % calculate memory cost
        memory = length(P(:))*8;
        Dhamm = hammingDist(B2, B1);
                
    case 'Kernel-ITQ'
        addpath('./ITQ');
        tic;
        [pc, l] = eigs(cov(param.KTrain), bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
        Y1 = param.KTrain * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
        % ITQ
        [B, R] = ITQ(Y1, 50);   %50Ϊ������RΪ���50�κ�õ�����ת����
        B1 = compactbit(B);
        training_time = toc + param.anchor_traintime;
        P = pc*R;
        tic;
        Y2 = param.KTest*P;      %�������ɷֽ���ͶӰ���������ݵ��ٽ�����ת�任
        B2 = compactbit(Y2>0);
        coding_time = toc + param.anchor_testtime;
        % compute Hamming metric and compute recall precision
        memory = length(P(:))*8;
        Dhamm = hammingDist(B2, B1);
        
    case 'Kernel-LSH'
        tic;
        W = randn(param.num_anchor, bit);
        Y1 = (param.KTrain*W >= 0);    %ԭ������Ļ�����ֵ��Ϊ0������0����Ϊ1��С��0����Ϊ0
        B1 = compactbit(Y1);
        training_time = toc + param.anchor_traintime;
        tic;
        Y2 = (param.KTest*W >= 0);
        B2 = compactbit(Y2);
        coding_time = toc + param.anchor_testtime;
        Dhamm = hammingDist(B2, B1);
        memory = length(W(:))*8;
        
        
    case 'BPBC'
        addpath('./BPBC');
        X_train = (param.X(:,1:num_train));
        X_test = (param.X(:,num_train+1:end));
        n = param.n;
        m = ceil(dim/n);
        r = param.r;
        clear param;
        
        tic;
        %--1D to 2D
        A1 = reshape(X_train,n,m,num_train);
		clear X_train;
        [R1, R2] =  BilinearITQ_low(A1, r, r, 50);
%       Y1 = reshape(Y1,m*r,num_train);
        B1 = logical(zeros(num_train, r*r));
        for i=1:num_train
            B1(i,:) = reshape(R1'*A1(:,:,i)*R2>0, 1, r*r);
        end
        B1 = compactbit(B1(1:num_train,:));
        training_time = toc
		clear A1;
        
        tic;
        A2 = reshape(X_test,n,m,num_test);
        clear X_test;
        B2 = logical(zeros(num_test, r*r));
        for i=1:num_test
            B2(i,:) = reshape(R1'*A2(:,:,i)*R2>0, 1, r*r);
        end
        B2 =  compactbit(B2);
        coding_time = toc
        
        memory = (length(R1(:))+length(R2(:)))*8;
        Dhamm = hammingDist(B2, B1);


    case 'CBE-opt'
        addpath('./CBE-opt');
        tic;
        para.bit = bit;
        para.iter = 10;
        train_size = min(num_train, 5000);
        [~, model] = circulant_learning(double(X_train(1:train_size, :)), para);
        B1 = CBE_prediction(model, X_train);
        if (para.bit < dim)
            B1 = B1 (:, 1:bit);
        end
        B1 = compactbit(B1>0);
        training_time = toc
 
        tic;
        B2 = CBE_prediction(model, X_test);
        B2 = B2 (:, 1:bit);
        B2 = compactbit(B2>0);
        coding_time = toc
        memory = (length(model.bernoulli)+length(model.r(:)))*8;
        Dhamm = hammingDist(B2, B1);
        
    case 'SP'
        % train
        tic;
        R = SP(double(X_train), bit, 0.9, 50);
        B1 = compactbit(X_train*R' >=0); 
        training_time = toc
                
        % coding
        tic;
        B2 = compactbit(X_test*R' >= 0);
        coding_time = toc
        memory = length(R(:))*8;
        Dhamm = hammingDist(B2, B1);

    case 'CCA-ITQ'
        addpath('./CCA');
        X_train = (param.X(:,1:num_train))';
        X_test = (param.X(:,num_train+1:end))';
        label = param.label;
        clear param;
        
        tic;
        [a, b, c] = unique(label);
        YY = zeros(num_train,length(a));
        for i=1:num_train
            YY(i,c(i)) = 1;
        end
%         bit1 = 40;
%         bit2 = bit-bit1;
        bit1 = bit;
        bit2 = 0;
        [eigenvector,r] = cca(X_train, YY, 0.0001); % this computes CCA projections
        eigenvector = eigenvector(:,1:bit1)*diag(r(1:bit1)); % this performs a scaling using eigenvalues
        Y1 = X_train*eigenvector; % final projection to otain embedding E
        YY1 = randn(num_train,bit2)>0;
        [B1, R] = ITQ(Y1,50);   %50Ϊ������RΪ���50�κ�õ�����ת����
        B1 = compactbit([B1,YY1]);
        training_time = toc
        P = eigenvector*R;
        tic;
        Y2 = X_test*P;
        YY2 =randn(num_test,bit2)>0;

        B2 = compactbit([Y2,YY2]>0);
        coding_time = toc
        memory = length(P(:))*8*4;
        Dhamm = hammingDist(B2, B1); 
 
    case 'KSH'
        addpath('./KSH');
        X_train = (param.X(:,1:num_train))';
        X_test = (param.X(:,num_train+1:end))';
        label = param.label;
        clear param.label
        
        tic;
        num_anchor = 300;
        sample = randperm(num_train, num_anchor);%300��anchor��
        anchor = X_train(sample',:);
        KTrain = sqdist(X_train',anchor');
        sigma = mean(mean(KTrain,2));
        KTrain = exp(-KTrain/(2*sigma));
        mvec = mean(KTrain);
        KTrain = KTrain-repmat(mvec,num_train,1);

        label_index = 1:1000;
        trn = 1000;
        % pairwise label matrix
        trngnd = label(label_index');
        temp = repmat(trngnd,1,trn)-repmat(trngnd',trn,1);
        S0 = -ones(trn,trn);
        tep = find(temp == 0);
        S0(tep) = 1;
        clear temp;
        clear tep;
        S = bit*S0;

        % projection optimization
        KK = KTrain(label_index',:);
        RM = KK'*KK; 
        A1 = zeros(num_anchor, bit);
        flag = zeros(1,bit);
        for rr = 1:bit
            [rr]
            if rr > 1
                S = S-y*y';
            end

            LM = KK'*S*KK;
            [U,V] = eig(LM,RM);
            eigenvalue = diag(V)';
            [eigenvalue,order] = sort(eigenvalue,'descend');
            A1(:,rr) = U(:,order(1));
            tep = A1(:,rr)'*RM*A1(:,rr);
            A1(:,rr) = sqrt(trn/tep)*A1(:,rr);
            clear U;    
            clear V;
            clear eigenvalue; 
            clear order; 
            clear tep;  

            [get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
            y = double(KK*A1(:,rr)>0);
            ind = find(y <= 0);
            y(ind) = -1;
            clear ind;
            y1 = double(KK*get_vec>0);
            ind = find(y1 <= 0);
            y1(ind) = -1;
            clear ind;
            if y1'*S*y1 > y'*S*y
                flag(rr) = 1;
                A1(:,rr) = get_vec;
                y = y1;
            end
        end

        % encoding
        B1 = compactbit(single(A1'*KTrain' > 0)');
        % tep = find(Y<=0);
        % Y(tep) = 0;
        training_time = toc
        % save ksh_48 Y A1 anchor mvec sigma;
        clear tep; 
        clear get_vec;
        clear y;
        clear y1;
        clear S;
        clear KK;
        clear LM;
        clear RM;


        % load ksh_48;
        % encoding
        tic;
        KTest = sqdist(X_test',anchor');
        KTest = exp(-KTest/(2*sigma));
        KTest = KTest-repmat(mvec,num_test,1);
        B2 = compactbit(single(A1'*KTest' > 0)');
        coding_time = toc
        memory = (length(anchor(:))+length(mvec)+length(A1(:))+1)*8;
        Dhamm = hammingDist(B2,B1);

    case  'PCAH'
        [pc, l] = eigs(cov(X_train),bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
        tic;
        X = X * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
        Y = zeros(size(X));
        Y(X>=0) = 1;
        training_time = toc;
        Y = compactbit(Y>0); 
        % compute Hamming metric and compute recall precision
    case 'CCA'
        tic;
        [a, b, c] = unique(label);
        YY = zeros(num_train,length(a));
        for i=1:num_train
            YY(i,c(i)) = 1;
        end
        [eigenvector,r] = cca(X_train, YY, 0.0001); % this computes CCA projections
        eigenvector = eigenvector(:,1:bit)*diag(r(1:bit)); % this performs a scaling using eigenvalues
        Y1 = X_train*eigenvector; % final projection to otain embedding E
        B1 = compactbit(Y1>0);
        training_time = toc
        tic;
        Y2 = X_test*eigenvector;
        B2 = compactbit(Y2>0);
        coding_time = toc
        memory = length(eigenvector(:))*8;
        Dhamm = hammingDist(B2, B1);

%     case 'PCA'
%         tic;
%         [pc, l] = eigs(cov(X_train),bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
%         training_time = toc
%         tic;
%         X = X * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
%         coding_time = toc
%         Dhamm = distMat(X_test', X_train');
%        
%     case 'LDA'
%         tic;
%         W= LDA(X',label, bit);
%         training_time = toc
%         tic;
%         Y = (X*W)';
%         coding_time = toc
%         Dhamm = distMat(Y(:,num_train+1:end),Y(:,1:num_train));
%     case 'PCA-LDA'
%         tic;
%         [pc, l] = eigs(cov(X_train),bit*bit);%��database�е���ݵ����Э�����Э������������ֽ��õ����ɷ�
%         X = X * pc;  %������ݵ������ɷ��Ͻ���ͶӰ
%         W= LDA(X',label, bit);
%         training_time = toc
%         tic;
%         Y = (X*W)';
%         coding_time = toc
%         Dhamm = distMat(Y(:,num_train+1:end),Y(:,1:num_train));     
%     case 'Euclidean Distance'
%         training_time = 0;
%         coding_time = 0;
%         Dhamm = distMat(X_test',X_train');
end

