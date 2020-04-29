%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation confusion matrix and ROC curve for SVMs are obtained by
% training on the HOG features extracted training+validation data with a 
% stratified 10-fold cross-validation and using the best hyper-parameters 
% obtained from SVMs_Training.m. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Import Scripts of HOG Features Extracted 
run('SVMs_HOG_Features.m')

%% Begin Traning SVMs with 10-fold Stratified Cross-Validation 
%Specify cross-validation partition
cvp = cvpartition(Train_label, 'KFold',10, 'Stratify', true);

%Preallocate cells to store the results of k-fold validation           
accuracy = zeros(length(k));

%Specify class names 
class_name = {'subject01','subject02','subject03','subject04','subject05',...
              'subject06','subject07','subject08','subject09','subject10'};
          
%For reproducibility
rng (1)

tic
for iter = 1: cvp.NumTestSets

    %Set up indexing
    train_idx = cvp.training(iter);
    test_idx  = cvp.test(iter);
    
    %Specify SVMs template using Gaussain and Linear
    %kernel functions
    t = templateSVM('KernelFunction', 'polynomial',...
                    'PolynomialOrder', 2,...
                    'BoxConstraint', 100,...
                    'Standardize', true);
    
    %Feed train features (training features + validation
    %features) into SVMs classifier with cross
    %validation and onevsone coding
    classifier = fitcecoc(std_Training_features(train_idx, :),...
                          Train_label(train_idx, :),...
                          'Learners', t,...
                          'ClassNames', class_name,...
                          'Coding', 'onevsone');
    
    %Obtain the predicted class labels
    pred = predict(classifier, std_Training_features);
    
    %Obtain the accuracy from the confusion matrix
    accuracy(iter) = sum(pred(test_idx) == Train_label(test_idx))/...
                     length(pred(test_idx));
                 
end                  
toc

%Take the average of the cross-validation accuracy
val_accuracy = mean(accuracy);

%Obtain the cross validation error using a
%stratified cross validation with training features
%and validation features
error = crossval('mcr',std_training_features, train_label,...
                 'Predfun', @classf, 'Stratify',train_label);

%% Test on the Validation Data             
%Obtain the predicted label 
[predict_label, score] = predict(classifier, std_validation_features);

%Convert the type to categorical 
predict_label = categorical(predict_label); 

Accuracy = sum(predict_label == val_label)/length(predict_label);
%% Plot confusion validation matrix with the best performing
figure 
plotconfusion(val_label, predict_label)
title({'Validation Confusion Matrix', 'SVMs'})

figure
hold on
[FP1, TP1, ~, AUC1] = perfcurve(val_label, score(:,1), 'subject01');
[FP2, TP2, ~, AUC2] = perfcurve(val_label, score(:,2), 'subject02');
[FP3, TP3, ~, AUC3] = perfcurve(val_label, score(:,3), 'subject03');
[FP4, TP4, ~, AUC4] = perfcurve(val_label, score(:,4), 'subject04');
[FP5, TP5, ~, AUC5] = perfcurve(val_label, score(:,5), 'subject05');
[FP6, TP6, ~, AUC6] = perfcurve(val_label, score(:,6), 'subject06');
[FP7, TP7, ~, AUC7] = perfcurve(val_label, score(:,7), 'subject07');
[FP8, TP8, ~, AUC8] = perfcurve(val_label, score(:,8), 'subject08');
[FP9, TP9, ~, AUC9] = perfcurve(val_label, score(:,9), 'subject09');
[FP10, TP10, ~, AUC10] = perfcurve(val_label, score(:,10), 'subject10');
plot(FP1, TP1)
plot(FP2, TP2)
plot(FP3, TP3)
plot(FP4, TP4)
plot(FP5, TP5)
plot(FP6, TP6)
plot(FP7, TP7)
plot(FP8, TP8)
plot(FP9, TP9)
plot(FP10, TP10)
legend('subject01','subject02','subject03','subject04','subject05',...
       'subject06','subject07','subject08','subject09','subject10')
title('SVMs Validation ROC Curve') 
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
hold off

