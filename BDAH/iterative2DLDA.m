function [R, L]=iterative2DLDA(Trainset, LabelTrain, p, q,r, c)


   %iterative 2DLDA, see Liang et al ."A note on two-dimensional linear discriminant analysis," Pattern Recogntion Letter
  % Trainset denotes the traning set  and each column is a data point
    %LabelTrain is the label of tranining set and is a colunm vector(such 1 denotes the 1th class and so on
   % r denotes the rows of images and c denotes the column of images
  %note that we set 10 iterations in our method
  % q is the right projected dimension and p is the left projected dimesnion
 % R is the right projected vectors and L is the left projected vectors. 

          [m,n]=size(Trainset);
          %--calculate M0 and Mi
          ClassNumber = max(LabelTrain);
          Mi = zeros(m, ClassNumber);
          Wi = zeros(ClassNumber,1);
          for i=1:ClassNumber
              temp=find(LabelTrain==i);
              Wi(i) = length(temp);
              Trainset1{i}=Trainset(:,temp');
              Mi(:,i)=mean(Trainset1{i},2);
          end
          M0 = mean(Trainset,2);
          
          %--initialize R
          R=[eye(q,q); zeros(c-q,q)];
          
          iter = 20;
          result = zeros(iter,1);
          Dw = zeros(iter,1);
          Db = zeros(iter,1);
         
          for j=1:iter
              twoDLDAiter=j
              % --obtian sb1 sw1
              Sb1=zeros(r,r);
              Sw1=zeros(r,r);
              for i=1:ClassNumber
                  for s=1:Wi(i)
                      Sw1=Sw1+(reshape(Trainset1{i}(:,s), r,c)-reshape(Mi(:,i), r,c))*(R*R')*(reshape(Trainset1{i}(:,s), r,c)-reshape(Mi(:,i), r,c))';
                  end
                  Sb1=Sb1+Wi(i)*(reshape(Mi(:,i), r,c)-reshape(M0, r,c))*(R*R')*(reshape(Mi(:,i), r,c)-reshape(M0, r,c))';
              end
              % --U=eigendecomposition(sb1,sw1);
              [U,S] =eig(pinv(Sw1)*Sb1);
              tt=diag(S);
              [B,IX]=sort(tt,'descend');
              U=U(:,IX);
                    
              L=U(:,1:p);
              Sb2=zeros(c,c);
              Sw2=zeros(c,c);
              for i=1:ClassNumber
                  for s=1:Wi(i)
                      Sw2=Sw2+(reshape(Trainset1{i}(:,s), r,c)-reshape(Mi(:,i), r,c))'*(L*L')*(reshape(Trainset1{i}(:,s), r,c)-reshape(Mi(:,i), r,c));
                  end
                  Sb2=Sb2+Wi(i)*(reshape(Mi(:,i), r,c)-reshape(M0, r,c))'*(L*L')*(reshape(Mi(:,i), r,c)-reshape(M0, r,c));
              end
              % U1=eigendecomposition(sb2,sw2);
              [U1,S1] =eig(pinv(Sw2)*Sb2);
              tt1=diag(S1);
              [~,IX1]=sort(tt1,'descend');
              U12=U1(:,IX1);
                  
              R=U12(:,1:q);

              for i=1:ClassNumber
                  for s=1:Wi(i)
                      DDw = L'*(reshape(Trainset1{i}(:,s), r,c)-reshape(Mi(:,i), r,c))*R;
                      Dw(j)=Dw(j)+sum(sum(DDw.*DDw));
                  end
                  DDb = L'*(reshape(Mi(:,i), r,c)-reshape(M0, r,c))*R;
                  Db(j)=Db(j)+Wi(i)*sum(sum(DDb.*DDb));
              end
              result(j) = Dw(j)/Db(j);
          end
figure;
subplot(1,3,1);
plot(1:iter, Dw);
xlabel('2DLDAiter');
ylabel('Dw');
subplot(1,3,2);
plot(1:iter, Db);
xlabel('2DLDAiter');
ylabel('Db');
subplot(1,3,3);
plot(1:iter, result);
xlabel('2DLDAiter');
ylabel('result');

                  