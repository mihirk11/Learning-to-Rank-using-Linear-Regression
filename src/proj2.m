%{
Learning to Rank using Linear Regression
Mihir Kulkarni
UB Person number: 50168610
Computer Science and Engineering
University at Buffalo
Buffalo, NY-14214
mihirdha@buffalo.edu
%}
load MyData.mat


%Saparate training testing validation sets
numPoints=size(XDouble,1);
randPerm=randperm(numPoints);

trainingData=XDouble(randPerm(1:round(numPoints*0.8)),:);
validationData=XDouble(randPerm(round(numPoints*0.8)+1:round(numPoints*0.9)),:);
trainingRlabel=rlabel(randPerm(1:round(numPoints*0.8)),:);
testingRlabel=rlabel(randPerm(round(numPoints*0.9)+1:numPoints),:);
validationRlabel=rlabel(randPerm(round(numPoints*0.8)+1:round(numPoints*0.9)),:);














%M1=3;%CHOOSE M1 HERE
validPer1Min=Inf(1);
gRow=1;
g=zeros(1000000,4);
for M1=2:10
     
        randM1=randperm(round(numPoints*0.8));
        mu=trainingData(randM1(1:M1),:);
        Sigma=zeros(size(trainingData,2),size(trainingData,2));
        Sigma=var(trainingData)*0.1;
        for i=1:size(trainingData,2)
            if (Sigma(1,i)<0.00001)
                Sigma(1,i)=0.1;
            end
        end   
        Sigma1=repmat(diag(Sigma),1,1,M1);

        trainingDataRows=size(trainingData);
        trainingDataRows=trainingDataRows(:,1);
        xMinusMu=repmat(trainingData,1,1,M1)-permute(repmat(mu,1,1,trainingDataRows),[3 2 1]);
        xMinusMuTranspose=permute(xMinusMu,[2 1 3]);

        phi1Training=zeros(trainingDataRows,M1);
        phi1Training(:,1)=1;
        for i=1:trainingDataRows
            for j=2:M1
                phi1Training(i,j)=exp((xMinusMu(i,:,j)*inv(Sigma1(:,:,j)))*xMinusMuTranspose(:,i,j)*(-0.5));%j-1??
            end
        end
        
        validationDataRows=size(validationData);
        validationDataRows=validationDataRows(:,1);
        xMinusMuValidation=repmat(validationData,1,1,M1)-permute(repmat(mu,1,1,validationDataRows),[3 2 1]);
        xMinusMuValidationTranspose=permute(xMinusMuValidation,[2 1 3]);

        phi1Validation=zeros(validationDataRows,M1);
        phi1Validation(:,1)=1;
        for i=1:validationDataRows
            for j=2:M1
                phi1Validation(i,j)=exp((xMinusMuValidation(i,:,j)*inv(Sigma1(:,:,j)))*xMinusMuValidationTranspose(:,i,j)*(-0.5));
            end
        end
        
        gcol=0;
     lambda1=0.1; % CHOOSE LAMBDA HERE
     while lambda1<0.6   
            w1=inv(lambda1*eye(M1,M1)+(transpose(phi1Training)*phi1Training))*(transpose(phi1Training)*trainingRlabel);
            trainPer1=sqrt(transpose(trainingRlabel-phi1Training*w1)*(trainingRlabel-phi1Training*w1)/trainingDataRows);
            validPer1=sqrt(transpose(validationRlabel-phi1Validation*w1)*(validationRlabel-phi1Validation*w1)/validationDataRows);
            if(validPer1Min>validPer1)
                validPer1Min=validPer1;
                M1Min=M1;
                lambda1Min=lambda1;
                trainPer1Min=trainPer1;
                w1Min=w1;
                phi1TrainingMin=phi1Training;
                phi1ValidationMin=phi1Validation;
                muMin=mu;
                Sigma1Min=Sigma1;
                
            end
            g(gRow,1)=M1;
            g(gRow,2)=lambda1;
            g(gRow,3)=trainPer1;
            g(gRow,4)=validPer1;
            g(gRow,5:5+M1-1)=w1;
            gRow=gRow+1;
            lambda1=lambda1+0.1;
      end
      lambda1=lambda1-0.1;%WE SHOULD IDEALLY PERFORM THIS ONLY WHEM WE HAVE INCREMENTED LAMBDA1 IN THE LOOP
end
lambda1=lambda1Min;
M1=M1Min;
w1=w1Min;
phi1Training=phi1TrainingMin;
phi1Validation=phi1ValidationMin;
mu=muMin;
Sigma1=Sigma1Min;


%FOR SYNTHETIC DATA
load synthetic.mat

XDouble2=x.';
rlabel2=t;

%Saparate training testing validation sets
numPoints2=size(XDouble2,1);
randPerm2=randperm(numPoints2);

trainingData2=XDouble2(randPerm2(1:round(numPoints2*0.8)),:);
validationData2=XDouble2(randPerm2(round(numPoints2*0.8)+1:numPoints2),:);
trainingRlabel2=rlabel2(randPerm2(1:round(numPoints2*0.8)),:);
testingRlabel2=rlabel2(randPerm2(round(numPoints2*0.9)+1:numPoints2),:);
validationRlabel2=rlabel2(randPerm2(round(numPoints2*0.8)+1:numPoints2),:);


%M2=3;        %CHOOSE M2 HERE
validPer2Min=Inf(1);
gRow2=1;
g2=zeros(1000000,4);
for M2=2:10
        randM2=randperm(round(numPoints2*0.8));
        mu2=trainingData2(randM2(1:M2),:);

        Sigma_s=zeros(size(trainingData2,2),size(trainingData2,2));
        Sigma_s=var(trainingData2)*0.1;
        for i=1:size(trainingData2,2)
            if (Sigma_s(1,i)<0.00001)
                Sigma_s(1,i)=0.1;
            end
        end   
        Sigma2=repmat(diag(Sigma_s),1,1,M2);

        trainingDataRows2=size(trainingData2);
        trainingDataRows2=trainingDataRows2(:,1);
        xMinusMu2=repmat(trainingData2,1,1,M2)-permute(repmat(mu2,1,1,trainingDataRows2),[3 2 1]);
        xMinusMuTranspose2=permute(xMinusMu2,[2 1 3]);



        phi1Training2=zeros(trainingDataRows2,M2);
        phi1Training2(:,1)=1;
        for i=1:trainingDataRows2
            for j=2:M2
                phi1Training2(i,j)=exp((xMinusMu2(i,:,j)*inv(Sigma2(:,:,j)))*xMinusMuTranspose2(:,i,j)*(-0.5));
            end
        end

        validationDataRows2=size(validationData2);
        validationDataRows2=validationDataRows2(:,1);

        xMinusMuValidation2=repmat(validationData2,1,1,M2)-permute(repmat(mu2,1,1,validationDataRows2),[3 2 1]);
        xMinusMuValidationTranspose2=permute(xMinusMuValidation2,[2 1 3]);

        phi1Validation2=zeros(validationDataRows2,M2);
        phi1Validation2(:,1)=1;
        for i=1:validationDataRows2
            for j=2:M2
                phi1Validation2(i,j)=exp((xMinusMuValidation2(i,:,j)*inv(Sigma2(:,:,j)))*xMinusMuValidationTranspose2(:,i,j)*(-0.5));
            end
        end
        
        lambda2=0.01;
        while lambda2<1  
            w2=inv(lambda2*eye(M2,M2)+(transpose(phi1Training2)*phi1Training2))*(transpose(phi1Training2)*trainingRlabel2);
            trainPer2=sqrt(transpose(trainingRlabel2-phi1Training2*w2)*(trainingRlabel2-phi1Training2*w2)/trainingDataRows2);    
            validPer2=sqrt(transpose(validationRlabel2-phi1Validation2*w2)*(validationRlabel2-phi1Validation2*w2)/validationDataRows2);
            if(validPer2Min>validPer2)
                validPer2Min=validPer2;
                M2Min=M2;
                lambda2Min=lambda2;
                trainPer2Min=trainPer2;
                w2Min=w2;
                phi1Training2Min=phi1Training2;
                phi1Validation2Min=phi1Validation2;
                mu2Min=mu2;
                Sigma2Min=Sigma2;
            end
            g2(gRow2,1)=M2;
            g2(gRow2,2)=lambda2;
            g2(gRow2,3)=trainPer2;
            g2(gRow2,4)=validPer2;
            g2(gRow2,5:5+M2-1)=w2;
            gRow2=gRow2+1;
            lambda2=lambda2+0.01;
      end
      lambda2=lambda2-0.01;
      
end
lambda2=lambda2Min;
M2=M2Min;
w2=w2Min;
phi1Training2=phi1Training2Min;
phi1Validation2=phi1Validation2Min;
Sigma2=Sigma2Min;
mu2=mu2Min;


%STOCASTIC GRADIENT DESCENT REAL WORLD TRAINING
numberOfIterations=10;
dw1=[];
w01f=rand(1,M1); %MAKE IT RANDOM
w01f=w01f*1000;
w01=w01f;
eta1f=1;
eta1=[];
dEd=zeros(size(trainingData,1),M1);
count=1;
T=40000;
prev_error=1;
for k=1:numberOfIterations
    for i=1:size(trainingData,1)
        
        %dEd(i,:)=-1*(trainingRlabel(i,:)-w01f*phi1Training(i,:).')*phi1Training(i,:)+lambda1/size(trainingData,1)*w01f;%IN FORMULAE TRANSPOSE IS OTHER WAY ROUND
        dEd(i,:)=-1*(trainingRlabel(i,:)-w01f*phi1Training(i,:).')*phi1Training(i,:)+lambda1*w01f;
        deltaW=-eta1f*dEd(i,:); 
        w01f=w01f+deltaW;
        
        %THIS IS CODE FOR BOLD DRIVER METHOD. BUT WE ARE NOT USING IT
        %error=(trainingRlabel(i,:)-phi1Training(i,:)*w01f.').'*(trainingRlabel(i,:)-phi1Training(i,:)*w01f.');
        %error=norm(w01f-w1.',2)/norm(w01-w1.',2);
        %{
        if(prev_error<error)
            eta1f=eta1f/2;
            
            %dw1(:,count)=0;
            %count=count-1;
            w01f=w01f-deltaW;
            i=i-1;
            if i==0
                i=size(trainingData,1);
                k=k-1;
                if k==0
                    k=k+1
                end
            end
            prev_error=error;
            continue;
        elseif(prev_error>error)
            eta1f=eta1f*1.05;
        end
        %}
        %eta1f=eta1f/(1+count*0.001/T);
        eta1(1,count)=eta1f;
        dw1(:,count)=deltaW.';
        count=count+1;
        %{
        norm(w01f-w1.',2)/norm(w01-w1.',2)
        if norm(w01f-w1.',2)/norm(w01-w1.',2)<0.1 && count>size(trainingData,1)
            break;
        end
        prev_error=error;
        %w01f
        %}
    end
    %{
    if norm(w01f-w1.',2)<0.1*norm(w01-w1.',2)
        break;
    end
    %}
end


%STOCASTIC GRADIENT DESCENT SYNTHETIC TRAINING
numberOfIterations2=10;
dw2=zeros(M2,numberOfIterations2*size(trainingData2,1));
w02f=rand(1,M2); %MAKE IT RANDOM
w02f=w02f*1000;
w02=w02f;
eta2f=1;
eta2=zeros(1,size(trainingData2,1)*numberOfIterations2);
dEd2=zeros(size(trainingData2,1),M2);
count2=1;

for k2=1:numberOfIterations2
    for i2=1:size(trainingData2,1)
        
        dEd2(i2,:)=-1*(trainingRlabel2(i2,:)-w02f*phi1Training2(i2,:).')*phi1Training2(i2,:)+lambda2*w02f;%IN FORMULAE TRANSPOSE IS OTHER WAY ROUND
        deltaW2=-eta2f*dEd2(i2,:);
        w02f=w02f+deltaW2;
        eta2(1,count2)=eta2f;
        
        dw2(:,count2)=deltaW2.';
        count2=count2+1;
        %w01f
        
    end
end

mu1=transpose(mu);
trainInd1=transpose(randPerm(1:55698));
validInd1=transpose(randPerm(55699:62660));

mu2=transpose(mu2);
trainInd2=transpose(randPerm2(1:round(numPoints2*0.8)));
validInd2=transpose(randPerm2(round(numPoints2*0.8)+1:numPoints2));
    
w01=w01.';
w02=w02.';

trainPer1=trainPer1Min;
validPer1=validPer1Min;

trainPer2=trainPer2Min;
validPer2=validPer2Min;
save('proj2.mat','M1','mu1','Sigma1','trainInd1','validInd1','w1','lambda1','trainPer1','validPer1','M2','mu2','Sigma2','trainInd2','validInd2','w2','lambda2','trainPer2','validPer2','w01','dw1','w02','dw2','eta1','eta2');
