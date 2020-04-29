%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  In this script, the best SVMs model is trained with a 10-fold stratified 
%  cross-validation using HOG features extracted training and validation data 
%  with the best hyper-parameters selected from the grid search in 
%  SVMs_Training.m, the averaged accuracy from the cross-validation is hence 
%  the validation accuracy.
%
%  Testing accuracy is obtained separatly in SVMs_Testing(Traning).m.
%
%  Total runtime for this script is 10204.943390 seconds.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Import Scripts of HOG Features Extracted 
run('SVMs_HOG_Features')

%% Begin Traning SVMs with 10-fold Stratified Cross-Validation 
%Define number of folds  
k = 10;

%Specify cross-validation partition
cvp = cvpartition(Train_label, 'KFold', k, 'Stratify', true);

%Preallocate cells to store the results of k-fold validation           
Accuracy = zeros(length(k));

%Specify the class names 
class_name = {'subject01','subject02','subject03','subject04','subject05',...
              'subject06','subject07','subject08','subject09','subject10'};

%Set up the template with the hyperparameters selected in SVMs training          
t = templateSVM('KernelFunction', 'polynomial',...
                'PolynomialOrder', 2,...
                'BoxConstraint', 100,...           
                'Standardize', true);
            
%For reproducibility           
rng (1) 

tic
%Train the SVMs for each fold             
for iter = 1: cvp.NumTestSets
    
    %Set up indexing 
    train_idx = cvp.training(iter);
    test_idx  = cvp.test(iter);
    
    %Feed the HOG features extracted training and validation data into SVMs 
    %classifier with the cross-validation and onevsone coding             
    classifier = fitcecoc(std_Training_features(train_idx, :),...
                          Train_label(train_idx, :),...
                         'Learners', t,...
                         'ClassNames', class_name,...
                         'Coding', 'onevsone');
    
    %Predict the class labels using the classifier                  
    pred = predict(classifier, std_Training_features);
    
    %Convert the type to categorical 
    pred = categorical(pred);
      
    %Obtain the accuracy of the predicted class labels against the class
    %labels of the testing set in the cvpartition
    Accuracy(iter) = sum(pred(test_idx) == Train_label(test_idx))/length(pred(test_idx));
    
end 
toc

%Obtain the average accuracy of 10-fold cross-validation 
val_accuracy = mean(Accuracy);

%Save the script into .mat format
save(strcat('SVMs_Final_',datestr(now,'yyyymmdd_HHMM'),'.mat'));
disp(strcat('done',datestr(now,'HH:MM')));