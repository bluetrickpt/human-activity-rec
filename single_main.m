clc;
clearvars;
close all;

addToPath =  genpath('stprtool');
addToPath = strcat(addToPath,genpath('libsvm-3.22'));
addToPath = strcat(addToPath,genpath('methods'));
addpath(addToPath);

nr_classes = 1;

data = load_dataset(nr_classes==1); %single class dataset

%normalize the data (x-mean)/std
data.X = zscore(data.X, 0 , 2);

%% Feature extraction
%FE: PCA, LDA and LLE
%FS: Filters

% =========== PCA ===============

%pca_model = pca(data.X);
%plot(pca_model.eigval); % By inspecting the graph, arround 160 eig values
%should retain most variance
%title('pca eig values');
%ylabel('eig values');

retain_variance = 0.95;
pca_model = pca(data.X, 1-retain_variance); 
pca_data = linproj(data, pca_model); %retains 104 with 0.95 variance

fprintf("%s%d%s%d%s%d%s\n", "PCA retaining ", retain_variance*100 , "% of variance has extracted ", size(pca_data.X, 1) , " from ", size(data.X,1),  " features");
% XR = pcarec(data.X, pca_model) %Reconstruction

pca_retained_dimensions = size(pca_data.X, 1);

% =========== LDA ===============
% lda_model = lda(data,pca_retained_dimensions); %Retaining the same number of features as pca did
% %duvida: Warning: Matrix is close to singular or badly scaled. Results may be inaccurate. RCOND =  1.647573e-27. 
% %os eig dão imaginários
% 
% cond(data.X) %2.8400e+20 <- Matrix singular
% cond(lda_model.Sw) %3.3809e+34
% %see: https://www.mathworks.com/matlabcentral/newsreader/view_thread/298397
% 
% lda_data =  linproj(data, lda_model);
% 
% fprintf("%s%d%s%d%s\n", "LDA has extracted ", size(lda_data.X,1), " from ", size(data.X,1) , " features" ); 

% =========== LLE ===============
%https://www.cs.nyu.edu/~roweis/lle/papers/lleintro.pdf
%lleK = pca_retained_dimensions; %For the nearest neighbors
%lleMaxDim = size(pca_data.X, 1); %At max, retaining the same number of features as pca did

%lle_data = lle(data.X, lleK, lleMaxDim);

%save('saves/single/lle_k104_maxdim104', 'lle_data');


load('saves/single/lle_k104_maxdim104');


fprintf("%s%d%s%d%s\n", "LLE has extracted ", size(lle_data.X,1), " from ", size(data.X,1) , " features" ); 


%% Feature selection
nr_features = size(data.X,1);
always_selected_features = 1:nr_features;

%=========== Feature correlation ===============
threshold = 0.9;

c = corrcoef(data.X');
[i,j] = find( abs(c) > threshold & c ~= 1.0 );

corr_data = data;
corr_data.X(j,:) = []; %c is transposed

fprintf("%s%f%s%d%s%d%s\n", "Feature correlation with ", threshold , " threshold selected ", size(corr_data.X,1), " from ", size(data.X,1) , " features" ); 

always_selected_features = intersect(always_selected_features,j);

%=========== Feature-class correlation ===============
threshold = 0.45;

c = corrcoef([data.X' data.y']);
idx = find(abs(c(end,1:end-1)) > threshold); 
%duvida: 
%the correlation of most features with class is arround 0.45: sort(abs(c(end,1:end-1))) 

class_corr_data = data;
class_corr_data.X = class_corr_data.X(idx,:); %c is transposed

fprintf("%s%f%s%d%s%d%s\n", "Class-Feature correlation with ", threshold , " threshold selected ", size(class_corr_data.X,1), " from ", size(data.X,1) , " features" ); 

always_selected_features = intersect(always_selected_features,idx);

%=========== AUC ===============
threshold = 0.85;

class_aucs = zeros(nr_classes, nr_features);

auc_values = zeros(1, nr_features);
for f=1:nr_features
    [~,~,~,auc_values(f)] = perfcurve(data.y, data.X(f,:),1);
end

%[~, idx] = sort(auc_values, 'descend');
idx = find(auc_values >= threshold);

auc_data = data;
auc_data.X = auc_data.X(idx, :); 

fprintf("%s%f%s%d%s%d%s\n", "AUC with ", threshold , " threshold selected ", size(auc_data.X,1), " from ", size(data.X,1) , " features" ); 

always_selected_features = intersect(always_selected_features,idx);

fprintf("%s%d\n", "Number of features that were selected in all previous feature selection methods: ", length(always_selected_features));

%% Assemble data structure to simplify classification
all_variants_data = [data pca_data  lle_data corr_data class_corr_data auc_data]; %lda_data
nr_variants = length(all_variants_data);

nr_classifiers = 3; %Change if more classifiers are used


dataset = repmat(struct('best_model', repmat(struct(), nr_classifiers, 1), 'best_model_auc', zeros(1,nr_classifiers),  'average_train_auc', zeros(1,nr_classifiers), 'average_test_auc', zeros(1,nr_classifiers)), nr_variants,1);

%% Cross-validation
folds=10; %10 folds
indices = crossvalind('Kfold', data.y, folds);
for i = 1:folds 

    test_idx = (indices == i); 
    train_idx = ~test_idx;
    
    fprintf(1, "%s%d\n", "Training dataset size: ", sum(train_idx));
    fprintf(1, "%s%d\n", "Test dataset size: ", sum(test_idx));
    
    
    for d=1:nr_variants
        
        
        train.X = all_variants_data(d).X(:,train_idx);
        train.y = all_variants_data(d).y(:,train_idx);
        test.X = all_variants_data(d).X(:,test_idx);
        test.y = all_variants_data(d).y(:,test_idx);
        
        if(d==1)
            fprintf(1, "%s%f\n", "Train balance (%): ", round(sum(train.y==1)/length(train.y) * 100, 2));
            fprintf(1, "%s%f\n", "Test balance (%): ", round(sum(test.y==1)/length(test.y) * 100,2));
        end
        

        %% Classifiers
        %k-NN, k-means, SVM

        %=========== kNN ===============
        classifier_idx = 1;
        %k=sqrt(length(train.y)); %The choice of K equal to the square root of the number of instances 
        %is an empirical rule-of-thumb popularized by the "Pattern Classification" book by Duda et al.
        %k=2*floor(k/2)+1; %rounds to the nearest odd number to avoid ties
        
        k=1; %k=1 gives good results
        [knn_model, train_auc, test_auc] = perform_knn(k, train, test, nr_classes);
        
        dataset(d).average_train_auc(classifier_idx) = dataset(d).average_train_auc(classifier_idx) + train_auc;
        dataset(d).average_test_auc(classifier_idx) = dataset(d).average_test_auc(classifier_idx) + test_auc;
        
        if(test_auc > dataset(d).best_model_auc(classifier_idx))
            dataset(d).best_model_auc(classifier_idx) = test_auc;
            %dataset(d).best_model(classifier_idx) = knn_model;
            
            for fn = fieldnames(knn_model)' %copy the model
               dataset(d).best_model(classifier_idx).(fn{1}) = knn_model.(fn{1});
            end
        end
        
        %=========== SVM ===============
        %libsvm uses one-against-one approach
        classifier_idx = 2;
        c_range = -3:3; 
        g_range = -5:2:5; %perform_svm uses log2 grid search
        
        [svm_model, train_auc, test_auc, c, g] = perform_svm(c_range, g_range , train, test, nr_classes, d);
       
        dataset(d).average_train_auc(classifier_idx) = dataset(d).average_train_auc(classifier_idx) + train_auc;
        dataset(d).average_test_auc(classifier_idx) = dataset(d).average_test_auc(classifier_idx) + test_auc;
        
        if(test_auc > dataset(d).best_model_auc(classifier_idx))
            dataset(d).best_model_auc(classifier_idx) = test_auc;
            
            for fn = fieldnames(svm_model)' %copy the model
               dataset(d).best_model(classifier_idx).(fn{1}) = svm_model.(fn{1});
            end
            
            dataset(d).best_model(classifier_idx).params = struct('c', c, 'g', g);
        end
        
        %=========== k-means ===============
        classifier_idx = 3;
        k=2; %each cluster is a class
        
        
        [auc, centers] = perform_kmeans(k, train, test); 
       
        %train and test have no meaning here
        dataset(d).average_train_auc(classifier_idx) = dataset(d).average_train_auc(classifier_idx) + auc;
        dataset(d).average_test_auc(classifier_idx) = dataset(d).average_test_auc(classifier_idx) + auc;
        
        if(auc > dataset(d).best_model_auc(classifier_idx))
            dataset(d).best_model_auc(classifier_idx) = auc;
            dataset(d).best_model(classifier_idx).params = struct('centers', centers);
        end
        
        
    end
end

for d=1:nr_variants
    for c=1:nr_classifiers
        dataset(d).average_train_auc(c) = dataset(d).average_train_auc(c)/folds;
        dataset(d).average_test_auc(c) = dataset(d).average_test_auc(c)/folds;
    end
end

save("saves/single/results", 'dataset');

%load("saves/single/best_results", 'dataset');
nElems = length(data.y);
print_results(dataset, nr_classes==1, nElems);

% Clean path 
rmpath(addToPath);


