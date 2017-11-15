function [best_model, saved_train_auc, best_test_auc, best_c, best_g] = perform_svm( c_range, g_range, train, test, nr_classes, d)
%PERFORM_SVM Summary of this function goes here
%   Detailed explanation goes here
%  


%     if(nr_classes==1)
%         load('saves/single/best_svm_models.mat'); %c 0:1:8; g -5:2:5
%     else
%         load('saves/multi/best_svm_models.mat');
%     end  
% 
%     best_model = best_svm_models(d).model;
%     saved_train_auc = best_svm_models(d).train_auc;
%     best_test_auc = best_svm_models(d).test_auc;
%     best_c = best_svm_models(d).c;
%     best_g = best_svm_models(d).g;

    saved_train_auc = []; best_test_auc=0; best_c=0; best_g=0; best_model = [];
    train_auc = zeros(1, nr_classes);
    test_auc = zeros(1, nr_classes);
    for log2c = c_range
        for log2g = g_range
            options = ['-s 0 -t 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; %c-svm with rbf kernel
            
            svm_model = svmtrain(train.y', train.X', options); 
            
            train_ypred = svmpredict(train.y', train.X', svm_model);
            test_ypred = svmpredict(test.y', test.X', svm_model);
            
            [~,~,~,~, train_auc, test_auc] = eval_classifier(train_ypred', train.y, test_ypred', test.y, nr_classes);    
              
            if(mean(test_auc)>mean(best_test_auc))
                best_test_auc=test_auc; best_c=log2c; best_g=log2g; best_model = svm_model; 
                saved_train_auc = train_auc;
            end
        end
    end
    

    
end

