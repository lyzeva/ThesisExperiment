%�ϼ���Train_Number�˵�Ŀ¼��ÿ��Ŀ¼���ĳ�˵�10������ͼ��,ȡǰn��ͼ����ѵ������
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
%�õ�ѵ�������ĸ���Train_Number
Train_Number = size(A,3);
%������ѵ�����������ľ�ֵ M
M=mean(A,3);

%��ͼ��ɢ������Gt
Gt=zeros(icol,icol);
for i = 1 : Train_Number
    temp = A(:,:,i)-M; 
    Gt =  Gt + temp'*temp;
end
Gt=Gt/Train_Number;
d = 10;
[V D] = eigs(Gt,d);
%�Уã��ռ�ͶӰͼ��
%V = orth(V);
ProjectedImages = zeros(irow,d);
for i = 1 : Train_Number
    ProjectedImages(:,:,i) = A(:,:,i)*V;     
end
%�ֿ�
%