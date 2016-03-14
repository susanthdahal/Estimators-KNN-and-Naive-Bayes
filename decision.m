function  decision()
    data=load('hw1boundary.mat');
    features=data.features;
    labels=data.labels;
    arr=[];
    termj=linspace(0,1,100);
    count=1;
    for i=1:100
        for j=1:100
            arr(count,1)=termj(i);
            arr(count,2)=termj(j);
            count=count+1;
        end
    end
    K=[1,5,15,25];
    for i=1:size(K,2)
       [new_accu, train_accu] = knn_classify(features,labels, arr,labels, K(i)) ;   
    end
end

