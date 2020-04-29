%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All neccesary data preparation is done in this script, such as importing
% images from the folders to image data store, split into training,
% validation, and testing data. A shuffle of the images before splitting is
% also done to ensure each dataset contains a random selection but equal of
% each subject. The split of training, validation, and testing images is
% 80:10:10. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CLear Work Space 
close all; clear all; clc 

%% Begin Data Preparation
%Import images to image datastore 
imagefolder = 'MIT-CBCL';
imds = imageDatastore(imagefolder, 'LabelSource', 'foldernames', 'IncludeSubFolders', true);
imds = shuffle(imds);

%Resize images to 64x64x1
imds_size = [64 64];
imds.ReadFcn = @(loc)imresize(imread(loc),imds_size);

%View number of images in each class
table = countEachLabel(imds);

%Define the number of images and their labels
num_imds  = numel(imds.Files); 
image_label = imds.Labels;

%Specify the number of classes 
num_class = 10;

%Split training and testing set wrt image preprocessing 
[Train_image, test_image] = splitEachLabel(imds, 0.9, 'randomized');
[train_image, val_image] = splitEachLabel(Train_image, 0.9, 'randomized');

%Define the labels for each images in training and teseting set
Train_label = Train_image.Labels;
train_label = train_image.Labels;
test_label  = test_image.Labels;
val_label   = val_image.Labels;

%View number of images in each class
table_Train = countEachLabel(Train_image);
table_train = countEachLabel(train_image);
table_test  = countEachLabel(test_image);
table_val   = countEachLabel(val_image);

%Specify number of training, validation, and testing images
Num_train = numel(Train_image.Files);
num_train = numel(train_image.Files);
num_test  = numel(test_image.Files);
num_val   = numel(val_image.Files);

