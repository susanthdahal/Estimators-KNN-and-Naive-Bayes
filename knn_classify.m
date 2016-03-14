function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, K)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  K: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CSCI 567: Machine Learning, Fall 2015, Homework 1
    %get_boundary(train_data, train_label, new_data, new_label, K)
    if size(new_data,1)==10000
        fprintf('\nDecision boundary for k=%d',K);
        [new_accu, train_accu]=get_boundary(train_data, train_label, new_data, new_label, K);
    else
        new_accu=get_accu(train_data, train_label, new_data, new_label, K);
        train_accu=get_accu(train_data, train_label, train_data, train_label, K);
    end
end

function [accuracy]=get_accu(train_data, train_label, new_data, new_label, K)
    length=size(new_data,1);
    classified=[];
    for i=1:length
        dist=zeros(size(train_data,1),1);
        for j=1:size(train_data,1)
            for k=1:size(train_data,2)
                if i==j
                    dist(j)=10000;
                end
            dist(j)=dist(j)+(new_data(i,k)-train_data(j,k)).^2;
            end
            dist(j)=sqrt(dist(j));
        end
        [sorted,indices]=sort(dist);
        labels=[];
        for k=1:K
            labels(k)=train_label(indices(k));
        end
        temp=unique(labels);
        if size(temp,2)==1
            classified(i)=temp(1);
        else
            [a,b]=hist(labels,unique(labels));
            [a,c]=max(a);
            classified(i)=b(c);
        end
    end
    count=0;
    for i=1:length
        if classified(i)==new_label(i)
            count=count+1;
        end
    end
    accuracy=count/length;
end

function [new_accu, train_accu]=get_boundary(train_data, train_label, new_data, new_label, K)
 
    length=size(new_data,1);
    classified=[];
    for i=1:length
        dist=[];
        for j=1:size(train_data,1)
            dist(j)=(new_data(i,1)-train_data(j,1)).^2;
            dist(j)=dist(j)+(new_data(i,2)-train_data(j,2)).^2;
            dist(j)=sqrt(dist(j));
        end
        [sorted,indices]=sort(dist);
        labels=[];
        for k=1:K
            labels(k)=train_label(indices(k));
        end
        temp=unique(labels);
        if size(temp,2)==1
            classified(i)=temp(1);
        else
            [a,b]=hist(labels,unique(labels));
            [a,c]=max(a);
            classified(i)=b(c);
        end
    end
    name= strcat('Decision boundary for k=', int2str(K));
    figure('Name',name,'NumberTitle','off');
    gscatter(new_data(:,1),new_data(:,2),classified,'br','xo');
    new_accu=1;
    train_accu=1;
end