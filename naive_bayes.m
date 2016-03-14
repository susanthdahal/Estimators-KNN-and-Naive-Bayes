function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)
% naive bayes classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data 
%
% CSCI 567: Machine Learning, Fall 2015, Homework 1

    fprintf('\nEntered into Naive bayes');
    length=size(train_data,1);
    breadth=size(train_data,2);
    cpt=containers.Map('KeyType','char','ValueType','any');
    step=1/length;
    labels=unique(train_label);
    for i=1:length
        label=char(train_label(i));
        if isKey(cpt,label)
            cpt(label)=cpt(label)+step;
        else
            cpt(label)=step;
        end
    end
    for i=1:breadth
        features=unique(train_data(:,i));
        for k=1:size(features,1)
            %tempkey=strcat('f#',char(i),'#',char(features(k)));
            for l=1:size(labels,1)
                count=0;
                label=char(labels(l));
                key=strcat('f#',char(i),'#',char(features(k)),'#',label);
                for j=1:length
                    if strcmp(char(train_data(j,i)),char(features(k))) && strcmp(char(train_label(j)),label)
                        count=count+1;
                    end
                end
                cpt(key)=count/length/cpt(label);
            end
            
        end
    end    
    new_accu=get_accuracy(train_data, train_label, new_data, new_label,cpt);
    train_accu=get_accuracy(train_data, train_label, train_data, train_label,cpt);

end
 function [accuracy]=get_accuracy(train_data, train_label, new_data, new_label,cpt)   
    new_length=size(new_data,1);
    new_breadth=size(new_data,2);
    labels=unique(train_label);
    classified=containers.Map('KeyType','int32','ValueType','any');
    for i=1:new_length
        max=0;
        for k=1:size(labels,1)
            label=char(labels(k));
            prob=1;
            if ~isKey(cpt,label)
                prob=0.1;
            else
                prob=cpt(label);
            end
            for j=1:new_breadth
                key=strcat('f#',char(j),'#',char(new_data(i,j)),'#',label);
                if ~isKey(cpt,key)
                    prob=prob*0.1;
                else
                    prob=prob*cpt(key);
                end
            end
            if prob>max
                max=prob;
                classified(i)=label;
            end
        end
    end
    
    count=0;
    for i=1:new_length
        if strcmp(classified(i),new_label(i))==1
            count=count+1;
        end
    end
    accuracy=count/new_length;
end