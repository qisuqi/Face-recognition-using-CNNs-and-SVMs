%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  The testing accuracy of SVMs is obtained in this script with the best
%  model trained in SVMs_Testing(Training).m.
%
%  A confusion matrix is then plotted to visualise the classifer. To 
%  evaluate the classifier, a ROC curve is plotted. The mis-classification
%  error is then calculated using the confusion matrix. 
%
%  Finally, a random selection of testing images are shown with their
%  predicted label obtained from the classifer. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Run Trained Best SVMs Model
load 'SVMs_Final_20200413_2338.mat'

%% Test on the Unseen Testing Data 
%Obtain the predicted label of the unseen testing set 
[predict_label, score] = predict(classifier, std_testing_features);

%Convert the type to categorical 
predict_label = categorical(predict_label); 

%Obtain the confusion matrix 
confmat = confusionmat(test_label, predict_label);
confmat = bsxfun(@rdivide, confmat, sum(confmat,2));

%Obtain the accuracy of the predicted class labels against the class labels
%of the testing set 
test_accuracy = mean(diag(confmat));

%Obtain the cross-validation error
error = crossval('mcr',std_Training_features, Train_label,...
                 'Predfun', @classf1, 'Stratify',Train_label);

%Visualise the confustion matrix                
figure 
plotconfusion(test_label, predict_label)
title({'Testing Confusion Matrix','SVMs'})

%Plot the ROC curve 
figure
hold on
[FP1, TP1, ~, AUC1]    = perfcurve(test_label, score(:,1), 'subject01');
[FP2, TP2, ~, AUC2]    = perfcurve(test_label, score(:,2), 'subject02');
[FP3, TP3, ~, AUC3]    = perfcurve(test_label, score(:,3), 'subject03');
[FP4, TP4, ~, AUC4]    = perfcurve(test_label, score(:,4), 'subject04');
[FP5, TP5, ~, AUC5]    = perfcurve(test_label, score(:,5), 'subject05');
[FP6, TP6, ~, AUC6]    = perfcurve(test_label, score(:,6), 'subject06');
[FP7, TP7, ~, AUC7]    = perfcurve(test_label, score(:,7), 'subject07');
[FP8, TP8, ~, AUC8]    = perfcurve(test_label, score(:,8), 'subject08');
[FP9, TP9, ~, AUC9]    = perfcurve(test_label, score(:,9), 'subject09');
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
title('SVMs Testing ROC Curve')  
xlabel('False Positive Rate') 
ylabel('True Positive Rate')
hold off

%Calculate the misclassification error using confusion matrix with
%FP+FN/Total
mis_error = [];

for m = 1: size(confmat,1)-1
    FP = confmat(:,m);
    FN = confmat(m,:);
    mis_error = (FP+FN)/sum(confmat,1);
end 

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


