%合计有Train_Number人的目录，每个目录存放某人的10幅人脸图像,取前n副图像做训练样本
n=5;
for i = 1 : class_Number
    topdir = int2str(i);
    topdir = strcat(TestDatabasePath,'\s',topdir);
    for j= 1 : n
     %for j= 1 :2 :9
        str = int2str(j);
        str = strcat(topdir,'\',str,'.pgm');
        img = imread(str);
        A(:,:,n*(i-1)+j) = img;  
    end
end
%得到训练样本的个数Train_Number
Train_Number = size(A,3);
%求所有训练样本向量的均值 M
M=mean(A,3);

%求图像散布矩阵Gt
Gt=zeros(icol,icol);
for i = 1 : Train_Number
    temp = A(:,:,i)-M; 
    Gt =  Gt + temp'*temp;
end
Gt=Gt/Train_Number;
d = 10;
[V D] = eigs(Gt,d);
%ＰＣＡ空间投影图像
%V = orth(V);
ProjectedImages = zeros(irow,d);
for i = 1 : Train_Number
    ProjectedImages(:,:,i) = A(:,:,i)*V;     
end
%分块
%