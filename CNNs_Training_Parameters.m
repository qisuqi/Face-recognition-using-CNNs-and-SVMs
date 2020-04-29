%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, a grid search is carried to adjust the hyper-parameters
% of CNNs using the best architecture obtained from CNNs_Training_Structure.m
% to train on the training data and test on the validation data. The 
% validation error is also calculated as a seconday criterion. The model 
% achieved the best validation accuracy and the validation error is 
% selected and its hyper-parameters are used for the testing stage in 
% CNNs_Testing.m.
%
% The hyper-parameters adjusted are mini batch size, initial learning rate,
% stochastic gradient decent momentum, epochs, and image preprocessing 
% techniques. 
%
% An early stopping trigger is also put in place to avoid the model
% overfitting.
% 
% A heatmap is also plotted to visualise the relationship between different
% image preprocessing techniques and their validation accuracy. 
%
% To visualise the confusion matrix and the ROC curve of the model that
% achieved the best validation accuracy and validation error is done
% separatly in CNNs_Validating.m. 
%
% Total runtime for this script is 75387.472412 seconds.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc 

%% Import Data Preparation 
run('CNNs_Data_Preparation.m')

%Specify image size 
image_size    = [32 32 1];

%Set up hyper-parameters for grid search 
minibatchsize = [16, 32, 64];
learning_rate = [0.00001, 0.001, 0.01, 0.1];
momentum      = [0, 0.5, 0.9];
epochs        = [30, 50, 100];
preprocessing = ["norm_val_image", "std_val_image", "zca_val_image"];

%Preallocate empty cells to store the results 
norm_Accuracy   = {}; std_Accuracy    = {}; zca_Accuracy    = {};
norm_Parameters = {}; std_Parameters  = {}; zca_Parameters  = {};

%% Begin Grid Search 
%For reproducibility 
rng(1)

tic
%Image preprocessing techniques 
for i = 1: length(preprocessing)
    %Mini batch size 
    for j = 1: length(minibatchsize)
        %Learning rate
        for k = 1: length(learning_rate)
            %Momentum
            for l = 1: length(momentum)
                %Numer of epochs
                for m = 1: length(epochs)
                    
                    %Normalised image
                    if preprocessing(i) == "norm_val_image"
                        
                        %Specify iterations per epochs
                        mini_batch_size = minibatchsize(j);
                        ite_per_epochs  = floor(num_train / mini_batch_size);
                        
                        %Define CNNs architecture
                        layers = [imageInputLayer(image_size)
                            
                                  convolution2dLayer(5, 4,...
                                                    'WeightsInitializer', 'glorot',...
                                                    'BiasInitializer', 'narrow-normal')
                                  batchNormalizationLayer
                                  reluLayer
                        
                                  maxPooling2dLayer(2, 'Stride', 2)
                        
                                  convolution2dLayer(7, 16)
                                  batchNormalizationLayer
                                  reluLayer
                        
                                  maxPooling2dLayer(2, 'Stride', 2)

                                  dropoutLayer(0.5)
                        
                                  fullyConnectedLayer(num_class)
                                  softmaxLayer
                                  classificationLayer];
                    
                          %Define training options
                          options = trainingOptions('sgdm',...
                                                    'MiniBatchSize', mini_batch_size,...
                                                    'ValidationData', {norm_val_image, val_label},...
                                                    'ValidationFrequency', ite_per_epochs,...
                                                    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,20),...
                                                    'InitialLearnRate', learning_rate(k),...
                                                    'Momentum', momentum(l),...
                                                    'MaxEpochs', epochs(m));
                    
                          %For reproducibility
                          rng(1)
                    
                          %Train the CNNs
                          [norm_net, norm_info]  = trainNetwork(norm_train_image, train_label, layers, options);
                    
                          %Obtain the predictions and the accuracies of
                          %the predictions
                          predict_norm_train  = classify(norm_net, norm_train_image);
                          predict_norm_val    = classify(norm_net, norm_val_image);
                          accuracy_norm_train = sum(predict_norm_train == train_label)/numel(train_label);
                          accuracy_norm_val   = sum(predict_norm_val == val_label)/numel(val_label);                        
                          
                          loss1 = (norm_info.ValidationLoss)';
                          loss1(isnan(loss1)) = 0;
                          norm_loss = mean(loss1);
                          
                          %Concatenate the parameters
                          norm_Parameters = [norm_Parameters; minibatchsize(j),learning_rate(k),...
                                                 momentum(l), epochs(m)];
                          
                          %Concatenate the accuracies
                          norm_Accuracy   = [norm_Accuracy; accuracy_norm_train, accuracy_norm_val,...
                                                 norm_loss];
                          
                          %Concatenate the parameters and the accuracies
                          norm_results = [norm_Accuracy norm_Parameters];
                          
                         
                    %Standardisation
                    elseif preprocessing(i) == "std_val_image"
                        
                            train_label = categorical(train_label);
                            val_label   = categorical(val_label);
                            
                            %Specify iterations per epochs
                            mini_batch_size = minibatchsize(j);
                            ite_per_epochs  = floor(num_train / mini_batch_size);
                        
                            %Define CNNs architecture
                            layers = [imageInputLayer(image_size);
                            
                                      convolution2dLayer(5, 4,...
                                                        'WeightsInitializer', 'glorot',...
                                                        'BiasInitializer', 'narrow-normal')
                                      batchNormalizationLayer
                                      reluLayer
                        
                                      maxPooling2dLayer(2, 'Stride', 2)
                        
                                      convolution2dLayer(7, 16)
                                      batchNormalizationLayer
                                      reluLayer
                        
                                      maxPooling2dLayer(2, 'Stride', 2)

                                      dropoutLayer(0.5)
                        
                                      fullyConnectedLayer(num_class)
                                      softmaxLayer
                                      classificationLayer];
                    
                             %Define training options
                             options = trainingOptions('sgdm',...
                                                       'MiniBatchSize', mini_batch_size,...
                                                       'ValidationData', {std_val_image, val_label},...
                                                       'ValidationFrequency', ite_per_epochs,...
                                                       'OutputFcn',@(info)stopIfAccuracyNotImproving(info,20),...    
                                                       'InitialLearnRate', learning_rate(k),...
                                                       'Momentum', momentum(l),...
                                                       'MaxEpochs', epochs(m));
                             %For reproducibility
                             rng(1)
                    
                             %Train the CNNs
                             [std_net, std_info] = trainNetwork(std_train_image, train_label, layers, options);
                             
                             %Obtain the predictions and the accuracies of
                             %the predictions
                             predict_std_train   = classify(std_net, std_train_image);
                             predict_std_val     = classify(std_net, std_val_image);
                             accuracy_std_train  = sum(predict_std_train == train_label)/numel(train_label);
                             accuracy_std_val    = sum(predict_std_val == val_label)/numel(val_label);
                             
                             loss2 = (std_info.ValidationLoss)';
                             loss2(isnan(loss2)) = 0;
                             std_loss = mean(loss2);
                             
                             %Concatenate the parameters
                             std_Parameters = [std_Parameters; minibatchsize(j),learning_rate(k),...
                                                    momentum(l), epochs(m)];
                             
                             %Concatenate the accuracies
                             std_Accuracy   = [std_Accuracy; accuracy_std_train, accuracy_std_val,...
                                                    std_loss];
                             
                             %Concatenate the parameters and the
                             %accuracies
                             std_results = [std_Accuracy std_Parameters];
                             
                   
                    else                        
                        
                        %Specify iterations per epochs
                        mini_batch_size = minibatchsize(j);
                        ite_per_epochs  = floor(num_train / mini_batch_size);
                        
                        %Define CNNs architecture
                        layers = [imageInputLayer(image_size);
                            
                                  convolution2dLayer(5, 4,...
                                                    'WeightsInitializer', 'glorot',...
                                                    'BiasInitializer', 'narrow-normal')
                                  batchNormalizationLayer
                                  reluLayer
                        
                                  maxPooling2dLayer(2, 'Stride', 2)
                        
                                  convolution2dLayer(7, 16)
                                  batchNormalizationLayer
                                  reluLayer
                        
                                  maxPooling2dLayer(2, 'Stride', 2)

                                  dropoutLayer(0.5)
                        
                                  fullyConnectedLayer(num_class)
                                  softmaxLayer
                                  classificationLayer];
                    
                         %Define training options
                         options = trainingOptions('sgdm',...
                                                   'MiniBatchSize', mini_batch_size,...
                                                   'ValidationData', {zca_val_image, val_label},...
                                                   'ValidationFrequency', ite_per_epochs,...
                                                   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,20),...
                                                   'InitialLearnRate', learning_rate(k),...
                                                   'Momentum', momentum(l),...
                                                   'MaxEpochs', epochs(m));
                         %For reproducibility
                         rng(1)
                    
                         %Train the CNNs
                         [zca_net, zca_info] = trainNetwork(zca_train_image, train_label, layers, options);
                         
                         %Obtain the predictions and the accuracies of
                         %the predictions
                         predict_zca_train   = classify(zca_net, zca_train_image);
                         predict_zca_val     = classify(zca_net, zca_val_image);
                         accuracy_zca_train  = sum(predict_zca_train == train_label)/numel(train_label);
                         accuracy_zca_val    = sum(predict_zca_val == val_label)/numel(val_label);                        
                         
                         loss3 = (zca_info.ValidationLoss)';
                         loss3(isnan(loss3)) = 0;
                         zca_loss = mean(loss3);
                         
                         %Concatenate the parameters
                         zca_Parameters = [zca_Parameters; minibatchsize(j),learning_rate(k),...
                                                momentum(l), epochs(m)];
                         
                         %Concatenate the accuracies
                         zca_Accuracy   = [zca_Accuracy; accuracy_zca_train, accuracy_zca_val,...
                                                zca_loss];
                                               
                         %Concatenate the parameters and the
                         %accuracies
                         zca_results = [zca_Accuracy zca_Parameters];                         
                    
                    end
                end
            end
        end
    end
end
toc
%% Format the results into a table and visualise the results 
%Mean-normalisation
norm_Results = array2table(norm_results);

%Split items in the arrays into individual columns 
for p = 1: size(norm_Results, 1)
    accus = cell2mat(norm_Results{p,1});
    paras = cell2mat(norm_Results{p,2});
    norm_Results{p,1} = {accus(1)};
    norm_Results{p,2} = {accus(2)};
    norm_Results{p,3} = {accus(3)};
    norm_Results{p,4} = {paras(1)};
    norm_Results{p,5} = {paras(2)};
    norm_Results{p,6} = {paras(3)};
    norm_Results{p,7} = {paras(4)};
    norm_Results{p,8} = {'Normalised'};
end 

%Define the column header 
norm_Results.Properties.VariableNames = {'Training_Accuracy', 'Validation_Accuracy',...
                                        'Loss','Mini_Batch_Size','Learning_Rate',...
                                        'Momentum', 'Epochs','Name'};  
%Standardisation                                 
std_Results = array2table(std_results);

%Split the items in the array into individual columns 
for q = 1: size(std_Results, 1)
    accus = cell2mat(std_Results{q,1});
    paras = cell2mat(std_Results{q,2});
    std_Results{q,1} = {accus(1)};
    std_Results{q,2} = {accus(2)};
    std_Results{q,3} = {accus(3)};
    std_Results{q,4} = {paras(1)};
    std_Results{q,5} = {paras(2)};
    std_Results{q,6} = {paras(3)};
    std_Results{q,7} = {paras(4)};
    std_Results{q,8} = {'Standardised'};
end 

%Define column header
std_Results.Properties.VariableNames = {'Training_Accuracy', 'Validation_Accuracy',...
                                        'Loss','Mini_Batch_Size','Learning_Rate',...
                                        'Momentum', 'Epochs','Name'};  

%ZCA                                
zca_Results = array2table(zca_results);

%Split the items in the array into individual columns 
for r = 1: size(zca_Results, 1)
    accus = cell2mat(zca_Results{r,1});
    paras = cell2mat(zca_Results{r,2});
    zca_Results{r,1} = {accus(1)};
    zca_Results{r,2} = {accus(2)};
    zca_Results{r,3} = {accus(3)};    
    zca_Results{r,4} = {paras(1)};
    zca_Results{r,5} = {paras(2)};
    zca_Results{r,6} = {paras(3)};
    zca_Results{r,7} = {paras(4)};
    zca_Results{r,8} = {'ZCA'};
end 

%Define column header
zca_Results.Properties.VariableNames = {'Training_Accuracy', 'Validation_Accuracy',...
                                        'Loss','Mini_Batch_Size','Learning_Rate',...
                                        'Momentum', 'Epochs','Name'};                               

% Obtain the best parameters with the best validation accuracies                                 
best_norm_val = max(cell2mat(norm_Results{:,2}));
best_std_val  = max(cell2mat(std_Results{:,2}));
best_zca_val  = max(cell2mat(zca_Results{:,2}));
best_norm_mdl = norm_Results(cell2mat(norm_Results.Validation_Accuracy) == best_norm_val, :);
best_std_mdl  = std_Results(cell2mat(std_Results.Validation_Accuracy) == best_std_val, :);
best_zca_mdl  = zca_Results(cell2mat(zca_Results.Validation_Accuracy) == best_zca_val, :);

%Concatenate both tables and obtain the best parameters overall
Results  = [norm_Results; std_Results; zca_Results];
Best_mdl = [best_norm_mdl; best_std_mdl; best_zca_mdl];

%Specify validation accuracy range into excellent, great, good and poor
excellent = Results(cell2mat(Results.Validation_Accuracy) >= 0.9, :);

great     = Results(cell2mat(Results.Validation_Accuracy) >= 0.7, :);
great     = great(cell2mat(great.Validation_Accuracy) < 0.9, :);

good      = Results(cell2mat(Results.Validation_Accuracy) >= 0.6, :);
good      = good(cell2mat(good.Validation_Accuracy) < 0.7,:);

poor      = Results(cell2mat(Results.Validation_Accuracy) <0.6, :);

excellent.Status = repmat(string('Excellent >= 90%'), height(excellent),1);
great.Status     = repmat(string('Great 90% > and >= 70%'), height(great),1);
good.Status      = repmat(string('Good 70% > and >= 60%'), height(good),1);
poor.Status      = repmat(string('Poor < 60%'), height(poor),1);

Final = [excellent; great; good; poor];

%Visualiase with a heatmap 
figure 
Heatmap = heatmap(Final, 'Name', 'Status');
title({'Validation Accuracy of Each','Image Processing Techniques'})


