clc;
clearvars;
close all;

addToPath =  genpath('../stprtool');
addToPath = strcat(addToPath,genpath('../libsvm-3.22'));
addpath(addToPath);

single_class_problem = true;


data = load_dataset(true);

%normalize the data (x-mean)/std
data.X = zscore(data.X, 0 , 2);



%% Feature extraction
%FE: PCA, LDA and LLE
%FS: Filters

% =========== PCA ===============

%pca_input_data = do_pca(input_data,retain_variance);
%fprintf("%s%d%s%d%s%d%s\n", "Pca extracted ", size(pca_input_data,2), " from ", size(input_data,2), " features, retaining ", retain_variance*100, "% of data variance");

%pca_model = pca(data.X);
%plot(pca_model.eigval); % By inspecting the graph, arround 160 eig values should be good

retain_variance = 0.95;
pca_model = pca(data.X, 1-retain_variance); 
pca_data = linproj(data, pca_model);

fprintf("%s%d%s%d%s%d%s\n", "PCA retaining ", retain_variance*100 , "% of variance has extracted ", size(pca_data.X, 1) , " from ", size(data.X,1),  " features");
% XR = pcarec(data.X, pca_model) %Reconstruction

pca_retained_dimensions = size(pca_data.X, 1);

% =========== LLE ===============
%lleK = size(pca_data.X, 1); %For the nearest neighbors
%lleMaxDim = size(pca_data.X, 1); %At max, retaining the same number of features as pca did

%lle_data = lle(data.X, lleK, lleMaxDim);

%save('saves/lle_k104_maxdim104', 'lle_data');
load('saves/lle_k104_maxdim104');


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
idx = find(abs(c(end,1:end-1)) > threshold); %duvida: the correlation of most features with class is arround 0.45... 

class_corr_data = data;
class_corr_data.X = class_corr_data.X(idx,:); %c is transposed

fprintf("%s%f%s%d%s%d%s\n", "Class-Feature correlation with ", threshold , " threshold selected ", size(class_corr_data.X,1), " from ", size(data.X,1) , " features" ); 

always_selected_features = intersect(always_selected_features,idx);

%=========== AUC ===============
threshold = 0.85;


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
all_variants_data = [data pca_data lda_data lle_data corr_data class_corr_data auc_data];
nr_variants = length(all_variants_data);

nr_classifiers=1;

dataset = repmat(struct('train', [], 'test', [], 'model', zeros(1,nr_classifiers),  'average_aucs', zeros(1,nr_classifiers)), nr_variants,1);

%% Cross-validation
[train_idx, test_idx] = crossvalind('HoldOut', data.y, 0.3); %30% is test data

fprintf(1, "%s%d\n", "Training dataset size: ", sum(train_idx));
fprintf(1, "%s%d\n", "Test dataset size: ", sum(test_idx));
    
%% Run from here when resuming execution
%load('saves/svm_param_runs');

best_svm_models=repmat(struct('model', [], 'c', [], 'g', [], 'auc', []), nr_variants,1);

for d=1:nr_variants
    dataset(d).train.X = all_variants_data(d).X(:,train_idx);
    dataset(d).train.y = all_variants_data(d).y(:,train_idx);
    dataset(d).test.X = all_variants_data(d).X(:,test_idx);
    dataset(d).test.y = all_variants_data(d).y(:,test_idx);

    %simplify notation
    train = dataset(d).train;
    test = dataset(d).test;

    best_auc=0; best_c=0; best_g=0; best_model = [];
    for log2c = 0:8
        for log2g = -5:2:5
            options = ['-s 0 -t 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; %c-svm with rbf kernel
            svm_model = svmtrain(train.y', train.X', options); 
            ypred = svmpredict(test.y', test.X', svm_model);

            [x,y] = perfcurve(test.y', ypred, 1);
            auc = trapz(x,y);

            if(auc>best_auc)
                best_auc=auc; best_c=log2c; best_g=log2g; best_model = svm_model;
            end
        end
    end

    best_svm_models(d).model = best_model;
    best_svm_models(d).c = best_c;
    best_svm_models(d).g = best_g;
    best_svm_models(d).auc = best_auc;
    
    
    %save('svm_param_runs', 'train_idx', 'test_idx', 'dataset', 'd', 'log2g', 'best_svm_models');

end


save('saves/best_svm_models', 'best_svm_models');

% Clean path 
rmpath(addToPath);

% Noite 1: c:-5:15 e g:-5:15; d manteve-se a 1, c chegou a 4. best_c=1,
% best_g =-5 e best_auc = 0.5252
% Não tenho tempo para testar tantas combinações...
% Mudei para log2c=0:8 e log2g=-5:5


