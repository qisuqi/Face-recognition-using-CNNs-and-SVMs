%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, a grid search is carried out to adjust the
% hyper-parameters of SVMs using HOG features extracted training and 
% validation data. Each model in the grid search is also trained
% with a 10-fold stratified cross-validation, the average accuracy of all
% 10 folds is then the validation accuracy. The misclassification error 
% during of cross-validation of each model is also calculated. 
%
% The results of all the hyper-parameters with their validation accuracy
% are appended into one table and the the model achieved the best validation
% accuracy is then selected to be used for testing stage. 
%
% The hyper-parameters in the grid search are kernel functions, box
% constraints, polynomial order for polynomial kernel function, and coding
% mode. 
%
% The range of selection for continuous hyper-parameters such as box
% constraints and polynomial order is inspired by the results from Baysian 
% Hyper-parameter Optimisation in SVMs_Opt.m
%
% A heatmap is also plotted to visualise the relationship between different
% kernel functions and their validation accuracy.
%
% To visualise the confusion matrix and the ROC curve of the model that
% achieved the best validation accuracy and misclassification error is done
% separatly in SVMs_Validating.m. 
%
% Total run time for this script is about 6 days. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space
close all; clear all; clc

%% Import Scripts of HOG Features Extracted 
run('SVMs_HOG_Features.m')

%% Begin Grid Search with 10-Fold Startified Cross-Validation 
%Define number of folds 
K = 10;

%Specify cross-validation partition
cvp = cvpartition(Train_label, 'KFold', K, 'Stratify', true);

%Specify class names 
class_name = {'subject01','subject02','subject03','subject04','subject05',...
              'subject06','subject07','subject08','subject09','subject10'};
          
%Set up parameters for SVMs template
box_cons   = [0.02, 20, 100, 400];
poly_order = [2, 3, 5];
kerl_fcn   = ["linear", "gaussian", "polynomial"];
mode       = ["onevsone", "onevsall"];

%Preallocate empty cells to store the results
accuracy1  = zeros(length(K)); accuracy2  = zeros(length(K)); 
accuracy3  = zeros(length(K)); accuracy4  = zeros(length(K));
Accuracy1  = {}; Accuracy2  = {}; Accuracy3  = {}; Accuracy4  = {};
Error1     = {}; Error2     = {}; Error3     = {}; Error4     = {};
Parameter1 = {}; Parameter2 = {}; Parameter3 = {}; Parameter4 = {};

%For reproducibility 
rng('default')
tic
for i = 1: length(kerl_fcn)
    for j = 1: length(box_cons)
        for k = 1: length(poly_order)
            for l = 1: length(mode)
            
                if kerl_fcn(i) == "polynomial"
                    
                    if mode(l) == "onevsone"
                        
                        for iter = 1: cvp.NumTestSets
                            
                            %Set up indexing 
                            train_idx = cvp.training(iter);
                            test_idx  = cvp.test(iter);
                            
                            %Specify SVMs template using Polynomial kernel
                            %functions 
                            t = templateSVM('KernelFunction', 'polynomial',...
                                            'BoxConstraint', box_cons(j),...
                                            'PolynomialOrder', poly_order(k),...
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
                            predict_label1 = predict(classifier, std_Training_features);  
                            
                            %Convert the type to categorical
                            predict_label1 = categorical(predict_label1);
                            
                            %Obtain the accuracy from the confusion matrix
                            accuracy1(iter) = sum(predict_label1(test_idx) == Train_label(test_idx))/...
                                              length(predict_label1(test_idx));
                                          
                            %Take the average of the cross-validation
                            %accuracy 
                            val_accuracy1 = mean(accuracy1);
                            
                            %Obtain the cross validation error using a
                            %stratified cross validation with training features
                            %and validation features
                            error1    = crossval('mcr',std_training_features, train_label,...
                                'Predfun', @classf, 'Stratify',train_label);
                       end    
                            %Store the results into the empty cells
                            Accuracy1  = [Accuracy1; val_accuracy1];
                            Error1     = [Error1; error1];
                            Parameter1 = [Parameter1; kerl_fcn(i), box_cons(j),...
                                          mode(l),poly_order(k)];                         
                                    
                    else

                        for iter = 1: cvp.NumTestSets

                            %Set up indexing
                            train_idx = cvp.training(iter);
                            test_idx  = cvp.test(iter);
                        
                            %Specify SVMs template using Polynomial kernel
                            %functions
                            t = templateSVM('KernelFunction', 'polynomial',...
                                            'BoxConstraint', box_cons(j),...
                                            'PolynomialOrder', poly_order(k),...
                                            'Standardize',true);
                            
                            %Feed train features (training features + validation
                            %features) into SVMs classifier with cross
                            %validation and onevsall coding
                            classifier = fitcecoc(std_Training_features(train_idx, :),...
                                                  Train_label(train_idx, :),...
                                                 'Learners', t,...
                                                 'ClassNames', class_name,...
                                                 'Coding', 'onevsall');
                            
                            %Obtain the predicted class labels
                            predict_label2 = predict(classifier,std_Training_features);                            
                            
                            %Obtain the accuracy from the confusion matrix
                            accuracy2(iter) = sum(predict_label2(test_idx) == Train_label(test_idx))/...
                                              length(predict_label2(test_idx));
                        
                            %Take the average of the cross-validation
                            %accuracy 
                            val_accuracy2 = mean(accuracy2);
                            
                            %Obtain the cross validation error using a
                            %stratified cross validation with training features
                            %and validation features
                            error2 = crossval('mcr',std_training_features, train_label,...
                                'Predfun', @classf, 'Stratify',train_label);

                        end     

                            %Store the results into the empty cells
                            Accuracy2  = [Accuracy2; val_accuracy2];
                            Error2     = [Error2; error2];
                            Parameter2 = [Parameter2; kerl_fcn(i), box_cons(j),...
                                          mode(l), poly_order(k)];
                    end 
                            
                else

                    if mode(l) == "onevsone"

                        for iter = 1: cvp.NumTestSets

                            %Set up indexing
                            train_idx = cvp.training(iter);
                            test_idx  = cvp.test(iter);

                            %Specify SVMs template using Gaussain and Linear
                            %kernel functions
                            t = templateSVM('KernelFunction', kerl_fcn(i),...
                                            'BoxConstraint', box_cons(j),...
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
                            predict_label3 = predict(classifier, std_Training_features);                            
                            
                            %Obtain the accuracy from the confusion matrix
                            accuracy3(iter) = sum(predict_label3(test_idx) == Train_label(test_idx))/...
                                              length(predict_label3(test_idx));

                            %Take the average of the cross-validation
                            %accuracy 
                            val_accuracy3 = mean(accuracy3);
                            
                            %Obtain the cross validation error using a
                            %stratified cross validation with training features
                            %and validation features
                            error3    = crossval('mcr',std_training_features, train_label,...
                                'Predfun', @classf, 'Stratify',train_label);
                           
                       end
                            %Store the results into the empty cells
                            Accuracy3  = [Accuracy3; val_accuracy3];
                            Error3     = [Error3; error3];
                            Parameter3 = [Parameter3; kerl_fcn(i), box_cons(j),...
                                          mode(l)];

                    else 

                        for iter = 1: cvp.NumTestSets

                            %Set up indexing
                            train_idx = cvp.training(iter);
                            test_idx  = cvp.test(iter);
                        
                            %Specify SVMs template using Gaussian and Linear
                            %kernel functions
                            t = templateSVM('KernelFunction', kerl_fcn(i),...
                                            'BoxConstraint', box_cons(j),...
                                            'Standardize',true);
                            
                            %Feed train features (training features + validation
                            %features) into SVMs classifier with cross
                            %validation and onevsall coding
                            classifier = fitcecoc(std_Training_features(train_idx, :),... 
                                                  Train_label(train_idx, :),...
                                                 'Learners', t,...
                                                 'ClassNames', class_name,...
                                                 'Coding', 'onevsall');
                            
                            %Obtain the predicted class labels
                            predict_label4 = predict(classifier, std_Training_features);

                            %Obtain the accuracy from the confusion matrix
                            accuracy4(iter) = sum(predict_label4(test_idx) == Train_label(test_idx))/...
                                              length(predict_label4(test_idx));

                            %Take the average of the cross-validation
                            %accuracy 
                            val_accuracy4 = mean(accuracy4);
                            
                            %Obtain the cross validation error using a
                            %stratified cross validation with training features
                            %and validation features
                            error4    = crossval('mcr',std_training_features, train_label,...
                                'Predfun', @classf, 'Stratify',train_label);
                       end     
                            %Store the results into the empty cells
                            Accuracy4  = [Accuracy4; val_accuracy4];
                            Error4     = [Error4; error4];
                            Parameter4 = [Parameter4; kerl_fcn(i), box_cons(j),...
                                          mode(l)];                        
                    end 
                end
            end 
        end 
    end 
end                                     
toc

%Append the results into a table 
results1 = array2table([Accuracy1 Error1 Parameter1; Accuracy2 Error2 Parameter2]);
results2 = array2table([Accuracy3 Error3 Parameter3; Accuracy4 Error4 Parameter4]); 

%Specify the polynomial is none for rbf and linear kernel functions 
for m = 1: size(results2, 1)   
    results2{m,6} = {'None'};
end 

%Append two tables and specify column header 
Results = [results1; results2];                   
Results.Properties.VariableNames = {'Validation_Accuracy', 'Error',...
                                    'Kernel_Function','Box_Constrains',...
                                    'Mode','Polynomial Order'};  
                                                                                   
%Obtain the best validation results                                
best_val = max(str2double(Results{:,1}));
best_mdl1 = Results(str2double(Results.Validation_Accuracy) == best_val, :);  

least_error = min(str2double(best_mdl1{:,2}));
final   = best_mdl1(str2double(best_mdl1.Error) == least_error, :);

%Specify validation accuracy range into excellent, great, good and poor
excellent = Results(str2double(Results.Validation_Accuracy) >= 0.9, :);

great     = Results(str2double(Results.Validation_Accuracy) >= 0.7, :);
great     = great(str2double(great.Validation_Accuracy) < 0.9, :);

good      = Results(str2double(Results.Validation_Accuracy) >= 0.6, :);
good      = good(str2double(good.Validation_Accuracy) < 0.7,:);

poor      = Results(str2double(Results.Validation_Accuracy) <0.6, :);

excellent.Status = repmat(string('Excellent >= 90%'), height(excellent),1);
great.Status     = repmat(string('Great 90% > and >= 70%'), height(great),1);
good.Status      = repmat(string('Good 70% > and >= 60%'), height(good),1);
poor.Status      = repmat(string('Poor < 60%'), height(poor),1);
                
Final = [excellent; great; good; poor];

%Visualiase with a heatmap 
figure 
Heatmap = heatmap(Final, 'Kernel_Function', 'Status');
title({'Validation Accuracy of Each','Kernel Functions'})



