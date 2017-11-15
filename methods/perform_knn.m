function [ knn_model,  train_auc, test_auc] = perform_knn(k, train, test, nr_classes)
%PERFORM_KNN Trains and evaluates a k-nn classifier.
%   For multi-class problems, returns arrays of results where the index of 
%   each value corresponds to the respective class.
    
    knn_model = knnrule(train, k);
    
    train_ypred = knnclass(train.X, knn_model);
    test_ypred = knnclass(test.X, knn_model);
    
    [~,~,~,~, train_auc, test_auc] = eval_classifier(train_ypred, train.y, test_ypred , test.y, nr_classes);    
    
%     for c=1:nr_classes
%         %train auc
%         [~,~,~,train_auc(c)] = perfcurve(train.y, train_ypred, c)
%         %plot(x,y)
%         %train_auc(c) = trapz(x,y); %area under the curve
% 
%         %test auc
%         [~,~,~,test_auc(c)] = perfcurve(test.y, test_ypred, c)
%         %figure; plot(x,y);
%         %test_auc(c) = trapz(x,y); %area under the curve      
%         
%     end
    
     
    %cerror(y_pred, test.y);
    %plot(x,y)
    %xlabel('false positives'); 
    %ylabel('false negatives');

    %[fp, fn] = roc(test.y, ypred);
    %figure; hold on; plotroc(test.y,ypred);

end

