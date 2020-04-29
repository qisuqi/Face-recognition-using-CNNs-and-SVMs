%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, the best hyper-parameters obtained from the training
% stage in CNNs_Training_Parameters.m are used to train on the
% training+validation data and test on the unseen testing data. 
%
% An early-stopping criteria is also specified to avoid the model
% overfitting, and the maximum numer of epochs is set to 100 to allow more
% training time since the size of training data has increased. 
%
% Then, a confusion matrix and a ROC curve are plotted to evaluate the
% classifer. 
%
% Misclassification error is also calculated using the confusion matrix. 
%
% Finally, a random selection of testing images are shown with their
% predicted label obtained from the classifer. 
% 
% Note: if a 100% validation accuracy is achieved, the model before this
% happened should be selected as part of the early-stopping implementation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc 

%% Import Data Preparation
run('CNNs_Data_Preparation.m')

%Specify image size 
image_size = [32 32 1];

%Specify mini batch size and iteration per epochs
mini_batch_size = 32;
ite_per_epochs  = floor(Num_train / mini_batch_size);

%Use the best performing network architecture for testing 
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

%Specify training options with the best hyper-parameters selected from the
%grid search       
options = trainingOptions('sgdm',...
                          'MiniBatchSize', mini_batch_size,...
                          'ValidationData', {std_val_image, val_label},...
                          'ValidationFrequency', ite_per_epochs,...
                          'InitialLearnRate', 0.1,...  
                          'Momentum', 0.9,...
                          'MaxEpochs', 100, ...
                          'OutputFcn',@(info)stopIfAccuracyNotImproving(info,10),...
                          'Plots', 'training-progress');

%For reproducibility                      
rng(1)    

%Train the network 
[net, info] = trainNetwork(std_Train_image, Train_label, layers, options);                 

%Predict the outcome of the network with the test images 
[predict_label, score] = classify(net, std_test_image);

%Obtain the confusion matrix 
conf_mat = confusionmat(test_label, predict_label);
conf_mat = bsxfun(@rdivide, conf_mat, sum(conf_mat,2));

%Obtain the testing accuracy 
accuracy = sum(predict_label == test_label)/numel(test_label);

%Plot the confusion matrix 
figure 
plotconfusion(test_label, predict_label)
title({'Testing Confusion Matrix','CNNs'})

%View a sample of predicted images against the labels of testing images 
idx = [1 5 10 15];
figure 
for l = 1:numel(idx)
    subplot(2,2,l)
    I = readimage(test_image, idx(l));
    label = predict_label(idx(l));
    imshow(I)
    title(char(label))
end 

%Calculate the misclassification error using confusion matrix with
%FP+FN/Total
mis_error = [];

for m = 1: size(conf_mat,1)-1
    FP = conf_mat(:,m);
    FN = conf_mat(m,:);
    mis_error = (FP+FN)/sum(conf_mat,1);
end 

%Plot ROC Curve
figure
hold on
[FP1, TP1, ~, AUC1] = perfcurve(test_label, score(:,1), 'subject01');
[FP2, TP2, ~, AUC2] = perfcurve(test_label, score(:,2), 'subject02');
[FP3, TP3, ~, AUC3] = perfcurve(test_label, score(:,3), 'subject03');
[FP4, TP4, ~, AUC4] = perfcurve(test_label, score(:,4), 'subject04');
[FP5, TP5, ~, AUC5] = perfcurve(test_label, score(:,5), 'subject05');
[FP6, TP6, ~, AUC6] = perfcurve(test_label, score(:,6), 'subject06');
[FP7, TP7, ~, AUC7] = perfcurve(test_label, score(:,7), 'subject07');
[FP8, TP8, ~, AUC8] = perfcurve(test_label, score(:,8), 'subject08');
[FP9, TP9, ~, AUC9] = perfcurve(test_label, score(:,9), 'subject09');
[FP10, TP10, ~, AUC10] = perfcurve(test_label, score(:,10), 'subject10');
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
title('CNNs Testing ROC Curve') 
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
hold off


