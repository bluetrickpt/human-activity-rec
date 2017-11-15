function [ balanced_dataset ] = balance_data( dataset )
%BALANCE_DATA Randomly downsamples the dataset to balance all classes
%   Detailed explanation goes here

    init_size = size(dataset.X,2);

    classes=unique(dataset.y);
    nr_classes = length(classes);
    class_occurrences = zeros(1, nr_classes);
    for c=1:nr_classes
        class_occurrences(c) = sum(dataset.y == c);
    end

    min_occurrences = min(class_occurrences);
    fprintf(1, "%s\n", "Minority class has " + min_occurrences + " points");

    nr_features = size(dataset.X,1);
    balanced_dataset.X = [];
    balanced_dataset.y = [];
    for c=1:nr_classes
        idx2include = find(dataset.y==c);
        rnd_idx = randperm(min_occurrences);
        
        balanced_dataset.X = [balanced_dataset.X dataset.X(:, idx2include(rnd_idx))];
        balanced_dataset.y = [balanced_dataset.y dataset.y(idx2include(rnd_idx))];
    end   

    final_size = size(balanced_dataset.X,2);
    
    fprintf(1, "%s\n", "Balancing classes reduced the dataset in " + (100-final_size/init_size*100) + "%");
    
end

