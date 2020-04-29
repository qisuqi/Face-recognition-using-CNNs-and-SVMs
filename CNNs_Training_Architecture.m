%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, a grid search is carried out to adjust the CNNs
% architecture using raw data and the architecture achieved the best
% validation accuracy is selected to be used in the grid search to adjust
% the hyper-parameters for CNNs in CNNs_Training_Parameters.m. 
%
% The best performing architecture is then visualised and analysed using
% analyzeNetwork. 
% 
% Total runtime for this script is 142267.363517 seconds.. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc 

%% Begin Grid Search on CNNs Architecture
%Import Data Preparation 
run('CNNs_Data_Preparation.m')

%Specify image size 
image_size   = [32 32 1];

%Set up parameters for grid search 
num_filters1 = [3, 5, 7];
filter_size1 = [2, 4, 6];
num_filters2 = [3, 5, 7];
filter_size2 = [4, 6, 8, 16];

%Preallocate empty cells to store the results 
Accuracy   = {};
Parameter1 = {};
Parameter2 = {};

%Begin grid search 
rng(1)
tic
for i = 1: length(num_filters1)
    for j = 1: length(filter_size1)
        for k = 1: length(num_filters2)
            for l = 1: length(filter_size2)               
        
                %Define the network architecture and iterate the number of
                %filters and the filter size 
                layers = [imageInputLayer(image_size); 
    
                          convolution2dLayer(num_filters1(i), filter_size1(j))
                          batchNormalizationLayer
                          reluLayer
          
                          maxPooling2dLayer(2, 'Stride', 2)
          
                          convolution2dLayer(num_filters2(k), filter_size2(l))
                          batchNormalizationLayer
                          reluLayer
          
                          maxPooling2dLayer(2, 'Stride', 2)

                          dropoutLayer(0.5)
          
                          fullyConnectedLayer(num_class)
                          softmaxLayer
                          classificationLayer];
              
                %Specify training options       
                options = trainingOptions('sgdm',...
                                          'MaxEpochs', 50, ...
                                          'ValidationData', val_image,...
                                          'ValidationFrequency', 30,....
                                          'InitialLearnRate', 0.00001);
                          
                %Train the network 
                net = trainNetwork(train_image, layers, options); 
        
                %Obtain the training and validation accuracy 
                predict_train   = classify(net, train_image);
                predict_val     = classify(net, val_image);
                accuracy_train  = sum(predict_train == train_label)/numel(train_label);
                accuracy_val    = sum(predict_val == val_label)/numel(val_label);
        
                %Concatenate accuracies and their parameters 
                Accuracy   = [Accuracy; accuracy_train, accuracy_val];
                Parameter1 = [Parameter1; num_filters1(i), filter_size1(j)];
                Parameter2 = [Parameter2; num_filters2(k), filter_size2(l)];
        
                results    = [Accuracy Parameter1 Parameter2];
            end 
        end 
    end 
end 
toc

%% Format the results into a table 
Results = array2table(results);

%Rearrange the table variables 
for m = 1: size(Results, 1)
    accus = cell2mat(Results{m,1});
    Results{m,1} = {accus(1)};
    Results{m,4} = {accus(2)};
end 

%Specify column headers
Results.Properties.VariableNames = {'Training_Accuracy', 'Cov1', 'Cov2',... 
                                    'Validation_Accuracy'}; 

%Obtain the best architecture                                
best_val = max(cell2mat(Results{:,4}));
best_mdl = Results(cell2mat(Results.Validation_Accuracy) == best_val, :);  
       
%% Visualise the best performing architecture
%Define the best performing architecture 
best_layers = [imageInputLayer(image_size, 'Name', 'input')

              convolution2dLayer(5, 4, 'Name', 'conv_1')
              batchNormalizationLayer('Name', 'bn_1')
              reluLayer('Name', 'relu_1')
              
              maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
              
              convolution2dLayer(7, 16, 'Name', 'conv_2')
              batchNormalizationLayer('Name', 'bn_2')
              reluLayer('Name', 'relu_2')
              
              maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
              
              dropoutLayer(0.5, 'Name', 'dropout')
              
              fullyConnectedLayer(num_class, 'Name', 'fc')
              softmaxLayer('Name', 'softmax')
              classificationLayer('Name', 'output')];

%Specify training options       
best_options = trainingOptions('sgdm',...
                               'MaxEpochs', 50, ...
                               'ValidationData', val_image,...
                               'ValidationFrequency', 30,...
                               'InitialLearnRate', 0.00001,...
                               'Plots', 'training-progress');
   
%Train the network with the best performing architecture                            
best_net = trainNetwork(train_image, best_layers, best_options);  

%Obtain the training and validation accuracy 
best_predict_train  = classify(best_net, train_image);
best_predict_val    = classify(best_net, val_image);
best_accuracy_train = sum(best_predict_train == train_label)/numel(train_label);
best_accuracy_val   = sum(best_predict_val == val_label)/numel(val_label);

%Analyse and visualise the best performing architecture                        
Best_architecture = layerGraph(best_layers);
analyzeNetwork(Best_architecture)
                                
