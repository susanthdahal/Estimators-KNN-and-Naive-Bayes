function main()

 fprintf('\nImages will be displayed later\nplease check the title of each image to know what it is');
kernel();


data=importdata('hw1nursery_train.data');
length=size(data,1);

train_data=[];
train_label=[];
for i=1:length
    c=char(data(i,1));
    c=strsplit(c,',');
    breadth=size(c,2);
    train_data=[train_data;c(1:breadth-1)];
    train_label=[train_label;c(breadth:breadth)];
end

data=importdata('hw1nursery_test.data');
new_data=[];
new_label=[];
length=size(data,1);
for i=1:length
    c=char(data(i,1));
    c=strsplit(c,',');
    breadth=size(c,2);
    new_data=[new_data;c(1:breadth-1)];
    new_label=[new_label;c(breadth:breadth)];
end
[new_accuracy,train_accuracy]=naive_bayes(train_data, train_label, new_data, new_label);
fprintf('\n\nNaive Bayes for Nursery data');
fprintf('\tNew accuracy=%f\t',new_accuracy);
fprintf('Train accuracy=%f\t',train_accuracy);
 data=importdata('hw1ttt_train.data');
length=size(data,1);

train_data=[];
train_label=[];
for i=1:length
    c=char(data(i,1));
    c=strsplit(c,',');
    breadth=size(c,2);
    train_data=[train_data;c(1:breadth-1)];
    train_label=[train_label;c(breadth:breadth)];
end

data=importdata('hw1ttt_test.data');
new_data=[];
new_label=[];
length=size(data,1);
for i=1:length
    c=char(data(i,1));
    c=strsplit(c,',');
    breadth=size(c,2);
    new_data=[new_data;c(1:breadth-1)];
    new_label=[new_label;c(breadth:breadth)];
end
[new_accuracy,train_accuracy]=naive_bayes(train_data, train_label,new_data, new_label);
fprintf('\n\nNaive Bayes for Tic-Tac-Toe data');
fprintf('\tNew accuracy=%f\t',new_accuracy);
fprintf('Train accuracy=%f\t',train_accuracy);

[train_data,train_label]=pre_processed(train_data,train_label);
[new_data,new_label]=pre_processed(new_data,new_label);
[new_accuracy,train_accuracy]=decision_Tree(train_data,train_label,new_data,new_label);


[train_data,train_label]=pre_process(train_data,train_label);
[new_data,new_label]=pre_process(new_data,new_label);
K=1:2:15;
for i=1:size(K,2)
    fprintf('\nk=%d\t',K(i));
    [new_accuracy,train_accuracy]=knn_classify(train_data, train_label, new_data, new_label, K(i));
    fprintf('new_accuracy=%f\t',new_accuracy);
    fprintf('train_accuracy=%f',train_accuracy);
end
decision();
end



function [new_table,new_label]=pre_process(table,label)

length=size(table,1);
breadth=size(table,2);
new_table=zeros(length,breadth*3);
for i=1:breadth
    ftable=unique(table(:,i));
    bin=size(ftable,1);
    for j=1:length
        temp=zeros(1,bin);
        for k=1:bin
            if strcmp(char(table(j,i)),char(ftable(k)))==1
                c=(bin*(i-1))+k;
                new_table(j,c)=1;
            end  
        end
    end
end
labels=unique(label);
new_label=[];
for i=1:length
    for j=1:size(labels,2)
        if strcmp(char(label(i)),char(labels(j)))==1
            new_label(i)=j;
        end
    end
end
end

function [new_accu, train_accu]=decision_Tree(train_data,train_label,new_data,new_label)
new_label=new_label';
train_label=train_label';
for i=1:20
    MinLeaf=i;
    SplitCriterion='gdi';
    fprintf('\n\n Splitcriterion =gdi, Minleaf=%d ',i);
    tree =ClassificationTree.fit(train_data, train_label, 'SplitCriterion', SplitCriterion, 'MinLeaf' ,MinLeaf,'Prune','off');
    result = predict(tree, new_data);
    new_accu = sum(result == new_label)/size(new_label,1);
    result = predict(tree, train_data);
    train_accu = sum(result == train_label)/size(train_label,1);
    fprintf('new accuracy =%f  ',new_accu);
    fprintf('train accuracy=%f  ',train_accu);
    SplitCriterion='deviance';
    fprintf('\n Splitcriterion =deviance, Minleaf=%d ',i);
    tree =ClassificationTree.fit(train_data, train_label, 'SplitCriterion', SplitCriterion, 'MinLeaf' ,MinLeaf,'Prune','off');
    result = predict(tree, new_data);
    new_accu = sum(result == new_label)/size(new_label,1);
    result = predict(tree, train_data);
    train_accu = sum(result == train_label)/size(train_label,1);
    fprintf('new accuracy =%f  ',new_accu);
    fprintf('train accuracy=%f  ',train_accu);
end
end


function [new_table,new_label]=pre_processed(table,label)

length=size(table,1);
breadth=size(table,2);
new_table=zeros(length,breadth);
for i=1:breadth
    ftable=unique(table(:,i));
    bin=size(ftable,1);
    for j=1:length
        for k=1:bin
            if strcmp(char(table(j,i)),char(ftable(k)))==1
               new_table(j,i)=k;
            end  
        end
    end
end
labels=unique(label);
new_label=[];
for i=1:length
    for j=1:size(labels,2)
        if strcmp(char(label(i)),char(labels(j)))==1
            new_label(i)=j;
        end
    end
end
end