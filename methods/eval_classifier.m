function [F1,precision,recall,accuracy, train_auc, test_auc] = eval_classifier (train_pred, ytrain, test_pred, ytest, nr_classes)


    accuracy = zeros(1,nr_classes);
    precision = zeros(1,nr_classes);
    recall = zeros(1,nr_classes);
    F1 = zeros(1,nr_classes);
    for c=1:nr_classes

        %Train eval (only auc)
        new_ytrain = ytrain;
        new_ytrain(ytrain==c) = 1;
        new_ytrain(ytrain~=c) = 0;

        new_train_pred = train_pred;
        new_train_pred(train_pred==c) = 1;
        new_train_pred(train_pred~=c) = 0;

        [~,~,~,train_auc(c)] = perfcurve(new_ytrain, new_train_pred, 1);

        %Test eval 
        new_ytest = ytest;
        new_ytest(ytest==c) = 1;
        new_ytest(ytest~=c) = 0;

        new_test_pred = test_pred;
        new_test_pred(test_pred==c) = 1;
        new_test_pred(test_pred~=c) = 0;

        [~,cm,~,~] = confusion(new_ytest, new_test_pred);

        tp = cm(1,1);
        tn = cm(2,2);
        fn = cm(1,2);
        fp = cm(2,1);

        accuracy(c) = (tp+fn) / (tp+tn+fp+fn);
        precision(c) = tp / (tp+fp);
        recall(c) = tp/(tp+fn);
        F1(c) = 2 * precision(c) * recall(c) / (precision(c) + recall(c));

        [~,~,~,test_auc(c)] = perfcurve(new_ytest, new_test_pred, 1);

    end
    accuracy = mean(accuracy);
    precision = mean(precision);
    recall = mean(recall);
    F1 = mean(F1);
   
end