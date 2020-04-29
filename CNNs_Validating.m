%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, a confusion and ROC curve is plotted using the results
% from CNNs_Training_Parameters.m of the model achieved the best validation
% accuracy and validation error. 
%
% Training progress is also plotted and an early stopping trigger is also 
% put in place. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Import Data Preparation
run('CNNs_Data_Preparation.m')

%Specify image size
image_size    = [32 32 1];

%Specify iterations per epochs
mini_batch_size = 32;
ite_per_epochs  = floor(num_train / mini_batch_size);

%Define CNNs architecture
layers = [imageInputLayer(image_size, 'Name', 'input')

          convolution2dLayer(5, 4, 'WeightsInitializer', 'glorot',...
                             'BiasInitializer', 'narrow-normal',...
                             'Name', 'conv_1')
          batchNormalizationLayer('Name', 'bn_1')
          reluLayer('Name', 'relu_1')

          maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

          convolution2dLayer(7, 16, 'Name', 'conv_2')
          batchNormalizationLayer('Name', 'bn_2')
          reluLayer('Name', 'relu_2')

          maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

          dropoutLayer(0.5)

          fullyConnectedLayer(num_class, 'Name', 'fc')
          softmaxLayer('Name', 'softmax')
          classificationLayer('Name', 'output')];

%Define training options
options = trainingOptions('sgdm',...
                          'MiniBatchSize', mini_batch_size,...
                          'ValidationData', {std_val_image, val_label},...
                          'ValidationFrequency', ite_per_epochs,...
                          'InitialLearnRate', 0.1,...
                          'Momentum', 0.9,...
                          'MaxEpochs', 30,...
                          'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10),...
                          'Plots', 'training-progress');
                      
%For reproducibility
rng(1)

%Train the CNNs
[net, info] = trainNetwork(std_train_image, train_label, layers, options);

% Obtain the predictions and the accuracies of
%the predictions
[train_predict_label, train_score] = classify(net, std_train_image);
[val_predict_label, val_score]   = classify(net, std_val_image);

%Obtain the validation accuracy 
train_accuracy = sum(train_predict_label == train_label)/numel(train_label);
val_accuracy   = sum(val_predict_label == val_label)/numel(val_label);

%Obtain the validation loss 
loss = (info.ValidationLoss)';
loss(isnan(loss)) = 0;
Loss = mean(loss);

%Plot confusion validation matrices with the best performing
%image preprocessing techinque 
figure
plotconfusion(val_label, val_predict_label)
title({'Validation Confusion Matrix', 'CNNs'})

%Plot ROC Curves
figure
hold on
[FP1, TP1, ~, AUC1]    = perfcurve(val_label, val_score(:,1), 'subject01');
[FP2, TP2, ~, AUC2]    = perfcurve(val_label, val_score(:,2), 'subject02');
[FP3, TP3, ~, AUC3]    = perfcurve(val_label, val_score(:,3), 'subject03');
[FP4, TP4, ~, AUC4]    = perfcurve(val_label, val_score(:,4), 'subject04');
[FP5, TP5, ~, AUC5]    = perfcurve(val_label, val_score(:,5), 'subject05');
[FP6, TP6, ~, AUC6]    = perfcurve(val_label, val_score(:,6), 'subject06');
[FP7, TP7, ~, AUC7]    = perfcurve(val_label, val_score(:,7), 'subject07');
[FP8, TP8, ~, AUC8]    = perfcurve(val_label, val_score(:,8), 'subject08');
[FP9, TP9, ~, AUC9]    = perfcurve(val_label, val_score(:,9), 'subject09');
[FP10, TP10, ~, AUC10] = perfcurve(val_label, val_score(:,10), 'subject10');
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
title({'Validation ROC Curve','CNNs'})  
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
hold off

