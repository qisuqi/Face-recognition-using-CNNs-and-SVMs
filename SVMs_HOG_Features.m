%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this script, HOG features are extracted from each training,
% validation, and testing data. Three image preprocessing techinques are
% then applied after the extraction to obtain the best image preprocessing 
% technique.  
%
% 4x4 pixels per cell with 9 histograms channels are used as the
% parameters to extract the HOG features. 
%
% A selection of HOG features extracted images from each dataset are then
% visualised. 
% 
% Finally, a simple SVMs classifier is used to train and test each image
% preprocessing technique and the technique achieved the best accuracy is
% used for Baysian Optimisation in SVMs_Opt.m, training in SVMs_Training.m,
% and testing in SVMs_Testing.m. This simple SVMs classifier will also
% serve as the SVMs baseline model. Both mean-normalisation and
% standardisation techinques achieved the same results, standardisation is
% chosen to be in line with CNNs and it achieved 52% accuracy. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Import Data Preparation
run('SVMs_Data_Preparation.m')

%Specify the whitening coefficient
epsilon = 0.1;
%% Training  + Validation Features  
TrainImage = readimage(Train_image, 1);          
[Train_hog, Train_vis] = extractHOGFeatures(TrainImage, 'CellSize',[4 4], 'NumBins',9);
Train_hog_feature_size = length(Train_hog);

%Preallocate empty cells to store the results 
Training_features      = zeros(Num_train, Train_hog_feature_size, 'single');
norm_Training_features = zeros(Num_train, Train_hog_feature_size, 'single');
std_Training_features  = zeros(Num_train, Train_hog_feature_size, 'single');
zca_Training_features  = zeros(Num_train, Train_hog_feature_size, 'single');

for i = 1: Num_train
    
    image = readimage(Train_image, i);
                                   
    Training_features(i, :) = extractHOGFeatures(image, 'CellSize',[4 4],...
                                                 'NumBins', 9);
    
    %Mean normalisation
    norm_Training_features(i, :) = Training_features(i, :) - mean2(Training_features(i, :));
    
    %Standardisation
    std_Training_features(i, :)  = norm_Training_features(i, :) ./std2(Training_features(i, :));
    
    %ZCA
    Training_Features(i, :) = Training_features(i, :)/255;
    norm_Training_Features(i, :) = Training_Features(i, :) - mean2(Training_Features(i, :));
    c(i, :) = cov(norm_Training_Features(i, :));
    [U(i, :), S(i, :), V(i, :)] = svd(c(i, :));
    
    zca_Training_features(i, :)  = U(i, :) .* diag(1/sqrt(diag(S(i, :))+epsilon))...
                                .* U(i, :).' .* norm_Training_Features(i, :); 
    
end 

%Visualise hog features extracted image
figure 
plot(Train_vis)
%% Training Features
trainmage = readimage(train_image, 1);          
[train_hog, train_vis] = extractHOGFeatures(trainmage, 'CellSize',[4 4], 'NumBins', 9);
train_hog_feature_size = length(train_hog);

%Preallocate empty cells to store the results 
training_features      = zeros(num_train, train_hog_feature_size, 'single');
norm_training_features = zeros(num_train, train_hog_feature_size, 'single');
std_training_features  = zeros(num_train, train_hog_feature_size, 'single');
zca_training_features  = zeros(num_train, train_hog_feature_size, 'single');

for j = 1: num_train
    
    image = readimage(train_image, j);
                                   
    training_features(j, :) = extractHOGFeatures(image, 'CellSize',[4 4],...
                                                 'NumBins', 9);
     
    %Mean normalisation
    norm_training_features(j, :) = training_features(j, :) - mean2(training_features(j, :));
    
    %Standardisation
    std_training_features(j, :)  = norm_training_features(j, :) ./std2(training_features(j, :));
    
    %ZCA
    trainingfeatures(j, :) = training_features(j, :)/255;
    norm_trainingfeatures(j, :) = trainingfeatures(j, :) - mean2(trainingfeatures(j, :));
    c1(j, :) = cov(norm_trainingfeatures(j, :));
    [U1(j, :), S1(j, :), V1(j, :)] = svd(c1(j, :));
    
    zca_training_features(j, :)  = U1(j, :) .* diag(1/sqrt(diag(S1(j, :))+epsilon))...
                                .* U(j, :).' .* norm_trainingfeatures(j, :); 
    
end 

%Visualise hog features extracted image
figure
plot(train_vis)
%% Validation Features
valimage = readimage(val_image, 1);          
[val_hog, val_vis] = extractHOGFeatures(valimage, 'CellSize',[4 4], 'NumBins', 9);
va_hog_feature_size = length(val_hog);

%Preallocate empty cells to store the results 
validation_features      = zeros(num_val, va_hog_feature_size, 'single');
norm_validation_features = zeros(num_val, va_hog_feature_size, 'single');
std_validation_features  = zeros(num_val, va_hog_feature_size, 'single');
zca_validation_features  = zeros(num_val, va_hog_feature_size, 'single');

for k = 1: num_val
    
    image = readimage(val_image, k);
                                   
    validation_features(k, :) = extractHOGFeatures(image, 'CellSize',[4 4],...
                                                 'NumBins', 9);
    
    %Mean normalisation
    norm_validation_features(k, :) = validation_features(k, :) - mean2(validation_features(k, :));
    
    %Standardisation 
    std_validation_features(k, :)  = norm_validation_features(k, :) ./std2(validation_features(k, :));
    
    %ZCA
    validationfeatures(k, :) = validation_features(k, :)/255;
    norm_validationfeatures(k, :) = validationfeatures(k, :) - mean2(validationfeatures(k, :));
    c2(k, :) = cov(norm_validationfeatures(k, :));
    [U2(k, :), S2(k, :), V2(k, :)] = svd(c2(k, :));
    
    zca_validation_features(k, :)  = U2(k, :) .* diag(1/sqrt(diag(S2(k, :))+epsilon))...
                                    .* U2(k, :).' .* norm_validationfeatures(k, :);
    
end 

%Visualise hog features extracted image
figure 
plot(val_vis)
%% Testing Features
testimage = readimage(test_image, 1);          
[test_hog, test_vis] = extractHOGFeatures(testimage, 'CellSize', [4 4], 'NumBins', 9);
test_hog_feature_size = length(test_hog);

%Preallocate empty cells to store the results 
testing_features      = zeros(num_test, test_hog_feature_size, 'single');
norm_testing_features = zeros(num_test, test_hog_feature_size, 'single');
std_testing_features  = zeros(num_test, test_hog_feature_size, 'single');
zca_testing_features  = zeros(num_test, test_hog_feature_size, 'single');

for l = 1: num_test
    
    image = readimage(test_image, l);
                                   
    testing_features(l, :) = extractHOGFeatures(image, 'CellSize', [4 4],...
                                                 'NumBins', 9);
    
    %Mean normalisation
    norm_testing_features(l, :) = testing_features(l, :) - mean2(testing_features(l, :));
    
    %Standardisation 
    std_testing_features(l, :)  = norm_testing_features(l, :) ./std2(testing_features(l, :));
    
    %ZCA
    testingfeatures(l, :) = testing_features(l, :)/255;
    norm_testingfeatures(l, :) = testingfeatures(l, :) - mean2(testingfeatures(l, :));
    c3(l, :) = cov(norm_testingfeatures(l, :));
    [U3(l, :), S3(l, :), V3(l, :)] = svd(c3(l, :));
    
    zca_testing_features(k, :)  = U3(l, :) .* diag(1/sqrt(diag(S3(l, :))+epsilon))...
                                .* U3(l, :).' .* norm_testingfeatures(l, :);
    
end 

%Visualise hog features extracted image
figure
plot(test_vis)
%% Train and Test the Classifier with the Features Extracted
norm_classfier = fitcecoc(norm_training_features, train_label);
norm_predict   = predict(norm_classfier, norm_testing_features);
norm_confmat   = confusionmat(test_label, norm_predict);
norm_accuracy  = mean(diag(norm_confmat));

std_classifier = fitcecoc(std_training_features, train_label);
std_predict    = predict(std_classifier, std_testing_features);
std_confmat    = confusionmat(test_label, std_predict);
std_accuracy   = mean(diag(std_confmat));

zca_classifier = fitcecoc(zca_training_features, train_label);
zca_predict    = predict(zca_classifier, zca_testing_features);
zca_confmat    = confusionmat(test_label, zca_predict);
zca_accuracy   = mean(diag(zca_confmat));

Results = [norm_accuracy, std_accuracy, zca_accuracy];

