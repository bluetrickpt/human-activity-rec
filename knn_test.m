clc;
clearvars;
close all;

addToPath =  genpath('stprtool');
addToPath = strcat(addToPath,genpath('libsvm-3.22'));
addToPath = strcat(addToPath,genpath('methods'));
addpath(addToPath);

%data = load_dataset(nr_classes==1); %single class dataset

trainFile = 'dataset/kaggle/train.csv';
testFile = 'dataset/kaggle/test.csv';

train_data = readtable(trainFile, 'ReadVariableNames', true);
test_data = readtable(testFile, 'ReadVariableNames', true);

train_input = table2array(train_data(:, 1:end-2)); %last 2 columns are subject_id and label
test_input = table2array(test_data(:, 1:end-2)); %last 2 columns are subject_id and label

train_output = string(table2cell(train_data(:, end))); %last column has the labels
test_output = string(table2cell(test_data(:, end))); %last column has the labels

train.X = train_input';
train.y = train_output';

test.X = test_input';
test.y = test_output';

classes = unique(train.y); %"LAYING"    "SITTING"    "STANDING"    "WALKING"    "WALKING_DOWNSTAIRS"    "WALKING_UPSTAIRS"
nr_classes=length(classes);
for c=1:nr_classes
    train.y(train.y==classes(c)) = c;
    test.y(test.y==classes(c)) = c;
end

train.y=double(train.y);
test.y=double(test.y);

%data.X = zscore(data.X, 0 , 2);
k=sqrt(length(train.y)); %The choice of K equal to the square root of the number of instances 
%is an empirical rule-of-thumb popularized by the "Pattern Classification" book by Duda et al.
k=2*floor(k/2)+1;
[knn_model, train_auc, test_auc] = perform_knn(k, train, test, nr_classes);



