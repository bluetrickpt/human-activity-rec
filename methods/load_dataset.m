function [data] = load_dataset(is_single_class_problem)

    trainFile = 'dataset/kaggle/train.csv';
    testFile = 'dataset/kaggle/test.csv';

    train_data = readtable(trainFile, 'ReadVariableNames', true);
    test_data = readtable(testFile, 'ReadVariableNames', true);

    train_input = table2array(train_data(:, 1:end-2)); %last 2 columns are subject_id and label
    test_input = table2array(test_data(:, 1:end-2)); %last 2 columns are subject_id and label

    data.X = [train_input; test_input]'; %append for single dataset

    train_output = string(table2cell(train_data(:, end))); %last column has the labels
    test_output = string(table2cell(test_data(:, end))); %last column has the labels

    data.y = [train_output; test_output]'; %append for single dataset


    if(is_single_class_problem == true)
        positive_indeces = (data.y =='WALKING');
        data.y(positive_indeces) = 1; %positive class
        data.y(~positive_indeces) = 0; %negative class
        
    else
        classes = unique(data.y) %"LAYING"    "SITTING"    "STANDING"    "WALKING"    "WALKING_DOWNSTAIRS"    "WALKING_UPSTAIRS"
        for c=1:length(classes)
            data.y(data.y==classes(c)) = c;
        end
    end
    
    data.y = double(data.y); %convert from string to double
end