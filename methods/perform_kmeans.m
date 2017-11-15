function [best_auc, centers] = perform_kmeans( k, train, test )
%PERFORM_KMEANS Summary of this function goes here
%   Detailed explanation goes here

    %train is used to form the centers, while in test we evaluate to which
    %center, each test observation belongs
    [idx, centers] = kmeans(train.X', k); 
    %k-means++ used for initial clusters and squared euclidean distance as measure

    %An observation belongs to the closest center. Let's populate ypred
    ypred = zeros(size(test.X,2), 1); 
    for p=1:size(test.X,2) %for each testing point
        ymin= -1; %there is no negative distances
        for c=1:k %k centers
            y = min(norm(test.X(:, p) - centers(c,:)'));
            if(y<ymin || ymin<0) %the distance is always positive
                ymin=y;
                ypred(p) = c;
            end
        end
    end
    
    %We don't know which cluster belongs to which class. 
    
    %So let's choose the best of the possible combinations in terms of AUC
    %note: this is inefficient!!! Checking the majority of occurrences in
    %each class would be better. However, due to poor results, some
    %peculiarities such as two or more clusters of the same class could occur.
    possibilities = perms(1:k);
    
    if k==2 %single class problem
        possibilities = possibilities-1;
        k=1; %to simplify
    end
    
    best_mean_auc = 0;
    best_auc = zeros(1,k);
    best_possibility =zeros(1,k);
    for p=1:size(possibilities,1) %for each possibility
        auc = zeros(1,k);
        for pos_class = 1:k %for each class has positive
            aux_ypred = ypred;
            aux_ypred(ypred==possibilities(p, pos_class)) = 1;
            aux_ypred(ypred~=possibilities(p, pos_class)) = 0;
            
            aux_testy = test.y;
            aux_testy(test.y==possibilities(p, pos_class)) = 1;
            aux_testy(test.y~=possibilities(p, pos_class)) = 0;
            
            [~,~,~,auc(pos_class)] = perfcurve(aux_testy, aux_ypred, 1);
        end
        
        if mean(auc) > best_mean_auc
            best_mean_auc = mean(auc);
            best_auc = auc;
            best_possibility = possibilities(p, :);
        end
    end
    
    if k>1 %multi class problem
        %Sort best_auc classes in accordance with best_possibility
        best_auc = best_auc(best_possibility);
    end   
    
end

