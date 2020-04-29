%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, the result of the CNNs baseline model is obtainied.
% The model is trained using raw training data, validated using raw
% validation data, and finally tested on unseen raw testing data. 
%
% The only parameters specified are the learning rate and the validation
% frequency. It was observed higher learning rate resulted in an error for
% undefined labels, and the validation frequency needs to be specified to
% obtain the validation accuracy during training stage. 
%
% The baseline model achieved an impressive 92.34% accuracy. However, the
% baseline model is restricted to a lower learning rate. Therefore, a grid
% search to adjust the hyper-parameters is still needed. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space
close all; clear all; clc 

%% Run Baseline Model
%Import Data Preparation 
run('CNNs_Data_Preparation.m')

%Specify image size 
image_size = [32 32 1];

%Define CNNs architecture using LeNet as the inspiration
layers = [imageInputLayer(image_size)

          convolution2dLayer(5, 6)

          averagePooling2dLayer(2, 'Stride', 2)

          convolution2dLayer(5, 16)

          averagePooling2dLayer(2, 'Stride', 2)         

          fullyConnectedLayer(num_class)
          softmaxLayer
          classificationLayer];

%Specify training options       
options = trainingOptions('sgdm',...
                          'InitialLearnRate', 0.00001,...
                          'ValidationData', val_image,...
                          'ValidationFrequency', 30,...
                          'Plots', 'training-progress');

%For reproducibility 
rng(1)

%Train the CNNs network 
net = trainNetwork(Train_image, layers, options);  

%Obtain the predicted label against validation data and the validation
%accuracy 
val_predict  = classify(net, val_image);
val_accuracy = sum(val_predict == val_label)/numel(val_label);

%Obtain the predicted label against unseen testing data and the testing 
%accuracy 
test_predict  = classify(net, test_image);
test_accuracy = sum(test_predict == test_label)/numel(test_label);