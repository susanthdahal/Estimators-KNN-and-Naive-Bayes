function kernel()
    data = load('hw1progde.mat');
    train_data= data.x_tr;
    fprintf('kernel density estimtion');
    width=[0.01 0.05 0.02 0.1 0.025];
    test_data=width/2:width:1;
    test_data=test_data';
    for i=1:size(width,2)
        result1=histo(train_data,test_data,width(i),0);
        result2=gauss(train_data,test_data,width(i),0);
        result3=epach(train_data,test_data,width(i),0);
    end
    output=[];
    test_data=data.x_te;
    for i=1:size(width,2)
        output=[output; {mise(test_data,width(i))}];
    end
    h=[];
    g=[];
    e=[];
    for i=1:size(width,2)
        temp=cell2mat(output(i));
        h(i)=temp(1);
        g(i)=temp(2);
        e(i)=temp(3);
    end
    name= ' Integrated square error vs width for histogram';
    figure('Name',name,'NumberTitle','off');
    bar(width,h);
    name= ' Integrated square error vs width for Gaussian kernel';
    figure('Name',name,'NumberTitle','off');
    bar(width,g);
    name= ' Integrated square error vs width for Epanechnikov kernel';
    figure('Name',name,'NumberTitle','off');
    bar(width,e);
end
    
function [result]=histo(train_data,test_data,width,flag)
    intervals=1/width;
    bin=zeros(intervals,1);
    
    for i=1:size(train_data,1)
        temp=ceil(train_data(i)/width);
        bin(temp)=bin(temp)+1;
    end
    for i=1:intervals
        bin(i)=bin(i)/(size(train_data,1)*width);
    end
    result=[];
    for i=1:size(test_data,1)
        temp=ceil(test_data(i)/width);
        if test_data(i)==0
            temp=1;
        end
        result(i)=bin(temp);
    end
    if(flag==0)
        name= strcat(' Histogram for h =',num2str(width));
        figure('Name',name,'NumberTitle','off');
        bar(test_data,result);
    end
end

function [result]= gauss(train_data,test_data,width,flag)
    y=[];
    for i =1:size(test_data,1)
        x=test_data(i);
        temp=0;
        for j=1:size(train_data,1)
            sub=(x-train_data(j))/width;
            temp=temp+ (1/(sqrt((2*pi))))*(exp(-1*(power(sub,2))/2));
        end
        y(i)=temp/(size(train_data,1)*width);
    end
    
    if(flag==0)
        name= strcat(' Gaussian kernel for h =',num2str(width));
        figure('Name',name,'NumberTitle','off');
        plot(test_data,y,'.');
    end
    result=y;
end
function [result]=epach(train_data,test_data,width,flag)
    y=[];
    for i =1:size(test_data,1)
        x=test_data(i);
        temp=0;
        for j=1:size(train_data,1)
            sub=(x-train_data(j))/width;
            if abs(sub) < 1
                temp=temp+ 3/4*(1-(sub*sub));
            end
        end
        y(i)=temp/(size(train_data,1)*width);
    end
    if flag==0
        name= strcat(' Epanechnikov kernel for h =',num2str(width));
        figure('Name',name,'NumberTitle','off');
        plot(test_data,y,'.');
    end
    result=y;
end

function [result]= mise(train_data,width)
    final={};
    ran=randperm(size(train_data,1));
    train_data=train_data(ran);
    for i=1:19
        inds=(i-1)*500+1;
        ends=inds+499;
        new_data=train_data(inds:ends,:);
        test_data=linspace(0,1,50);
        test_data=test_data';
        final{i,1}=histo(new_data,test_data,width,1);
        final{i,2}=gauss(new_data,test_data,width,1);
        final{i,3}=epach(new_data,test_data,width,1);
    end
    length=size(final,1);
    breadth=size(final,2);
    err=zeros(length,breadth);
    for i=1:breadth
        total=zeros(1,50);
        for j=1:length
            temp=cell2mat(final(j,i));
            total=total+temp;
        end 
        m=total/length;
        for j=1:length
            temp=cell2mat(final(j,i));
            temp=temp-m;
            temp=power(temp,2);
            total=sum(temp);
            err(j,i)=total;
        end
    end
    arr=zeros(1,breadth);
    for i =1:breadth
        arr(i)=sum(err(:,i))/length;
    end
    result=arr;
end
