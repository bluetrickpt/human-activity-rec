function [] = print_results( dataset, is_single_class_problem, dataset_size, nr_folds )
%SAVE_PRINT_RESULTS Summary of this function goes here
%   Detailed explanation goes here
    
    if(is_single_class_problem)
        filename="results/single_results";
    else
        filename="results/multi_results";
    end
    
    fileID = fopen(filename,'w+'); %to write
    
    classifiers = ["k-NN", "SVM", "k-means (k=nr_classes)"];
    header=[" ", " ", classifiers];
    for i=1:length(header)
        fprintf(fileID,"%s\t", header(i));
    end
    fprintf(fileID,"%s\n", "");
    
    dataset_names=["Full Dataset", "PCA", "LLE", "Feature Corr", "Class Corr", "AUC"];
    
    second_col=["avg_train_auc", "avg_test_auc", "max_test_auc"];
    
    for d=1:length(dataset)
        %three rows for each dataset
        first_row = [" ", second_col(1), mean(dataset(d).average_train_auc(1,:)), mean(dataset(d).average_train_auc(2,:)), mean(dataset(d).average_train_auc(3,:))];
        second_row = [dataset_names(d), second_col(2), mean(dataset(d).average_test_auc(1,:)), mean(dataset(d).average_test_auc(2,:)), mean(dataset(d).average_test_auc(3,:))];
        third_row = [" ", second_col(3), mean(dataset(d).best_model_auc(1,:)), mean(dataset(d).best_model_auc(2,:)), mean(dataset(d).best_model_auc(3,:))];
    
        dataset_line = [first_row; second_row; third_row];
       
        for r=1:size(dataset_line, 1)
            for c=1:length(dataset_line(r, :))
                fprintf(fileID, "%s\t", dataset_line(r,c));
            end
            fprintf(fileID, "%s\n", "");
        end
    end
    

    fprintf(fileID,"\n%s\n", "==================================");
    
    if(~is_single_class_problem)
        
        for c=1:length(classifiers)
            fprintf(fileID, "\n%s\n", "Average test auc per class for classifier " + classifiers(c));

            header = [" ", "C1", "C2", "C3", "C4", "C5", "C6" ];
            for i=1:length(header)
                fprintf(fileID,"%s\t", header(i));
            end
            fprintf(fileID,"%s\n", "");

            for d=1:length(dataset)
                row = [dataset_names(d), dataset(d).average_test_auc(c,:)];
                for i=1:length(row)
                    fprintf(fileID,"%s\t", row(i));
                end
                fprintf(fileID,"%s\n", "");
            end
        end
    end
    
    
    fprintf(fileID,"\n%s\n", "==================================");
    
    train_ratio = 0.7;
    if ~is_single_class_problem
        train_ratio=1-1/nr_folds;
    end
    
    fprintf(fileID, "%s%d%s\n", "Number of SV for each dataset: (dataset training size ~", round(train_ratio*dataset_size), ")");
    for d=1:length(dataset)
        
        best_c = 2^dataset(d).best_model(2).params.c;
        best_g = 2^dataset(d).best_model(2).params.g;
        fprintf(fileID, "%s", dataset_names(d) + " (c=" + best_c + ", g=" + best_g + "): ");
        
        nSV = dataset(d).best_model(2).totalSV;
        sv_all_ration = nSV/round(train_ratio*dataset_size)*100;
        fprintf(fileID, "%s\n", nSV + " (" + sv_all_ration + "% of the training data)");
    end
    fclose(fileID);
end

