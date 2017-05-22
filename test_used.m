
Get the recall & precision 8

tic;
disp('PCA-ITQ:');
Dhamm = hashCalculator(X', Groundtruth, bit, 'ITQ');
toc
[recall{1,1}, precision{1,1}, thresh{1,1}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

b1 = 5;
b2 = 128;
tic;
disp('BITQ:');
Dhamm = twoDBITQ(X, Groundtruth, 128, b1, length);
toc
[recall{1,1}, precision{1,1}, thresh{1,1}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

% tic;
disp('PCA-RR:');
[recall{1,2}, precision{1,2}] = hashCalculator(X', Groundtruth, bit, 'RR');
toc

tic;
disp('2DPCA-PQ');
[recall{1,2}, precision{1,2}, thresh{1,2}] = twoDHashPQ(X, Groundtruth, length, rank, bit);
toc


tic;
disp('2DPCA-2DITQ:');
Dhamm = twoDHash2DITQ(X, Groundtruth, length, rank);
toc
[recall{1,3}, precision{1,3}, thresh{1,3}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

tic;
disp('2DPCA-ITQ:');
Dhamm = twoDHashITQ(X, Groundtruth, length, rank);
toc
[recall{1,4}, precision{1,4}, thresh{1,4}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

tic;
disp('2DPCA:');
Dhamm = twoDHash(X, Groundtruth, length, rank);
toc
[recall{1,5}, precision{1,5}, thresh{1,5}]= recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

tic;
disp('LSH');
Dhamm = hashCalculator(X', Groundtruth, bit, 'LSH');
toc;
[recall{1,6}, precision{1,6}, thresh{1,6}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);

% tic;
% disp('PCAH');
% Dhamm = hashCalculator(X', Groundtruth, bit, 'PCAH');
toc
[recall{1,7}, precision{1,7}, thresh{1,7}] = recall_precision_kNN(Groundtruth, Dhamm, 10*NN, 1);


[recall{1,5}, precision{1,5}] = twoDHash3(XX, 28, 28, Groundtruth, length, rank);
length = 8;
rank = 7;
[recall{1,5}, precision{1,5}] = twoDHash2(XX, 28, 28, Groundtruth, length, rank);