%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All neccesary data preparation is done in this script, such as importing
% images from the folders to image data store, split into training,
% validation, and testing data. A shuffle of the images before splitting is
% also done to ensure each dataset contains a random selection but equal of
% each subject. The split of training, validation, and testing images is
% 80:10:10. 
%
% Next, three image preprocessing techniques are applied to each dataset
% namely mean normalisation, stardisation, and zca. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space
close all; clear all; clc 

%% Data Preparation 
%Import images into image datastore 
imagefolder = 'MIT-CBCL';
imds = imageDatastore(imagefolder, 'LabelSource', 'foldernames', 'IncludeSubFolders', true);

%View number of images in each class
table = countEachLabel(imds);

%Specify the number of images and their labels 
num_imds   = numel(imds.Files); 
imds_label = imds.Labels;

%View a sample of images in the database 
idx = 1: 10;
figure
for i = 1:10
    subplot(2,5,i)
    I= readimage(imds, idx(i));
    label = imds_label(idx(i));
    imshow(I, 'Border', 'tight')
    title(char(label))
end

%Shuffle images 
imds = shuffle(imds);

%Specify the number of classes 
num_class = 10;

%Resize images to 32x32x1
imds_size = [32 32];
imds.ReadFcn = @(loc)imresize(imread(loc),imds_size);

%Split training and testing set wrt image preprocessing 
[Train_image, test_image] = splitEachLabel(imds, 0.9, 'randomized');
[train_image, val_image]  = splitEachLabel(Train_image, 0.9, 'randomized');

%View number of images in each class
table_Train = countEachLabel(Train_image);
table_train = countEachLabel(train_image);
table_test  = countEachLabel(test_image);
table_val   = countEachLabel(val_image);

%Define the labels for each images in training and teseting set
Train_label = Train_image.Labels;
train_label = train_image.Labels;
test_label  = test_image.Labels;
val_label   = val_image.Labels;

%Specify number of training, validation, and testing images
Num_train = numel(Train_image.Files);
num_train = numel(train_image.Files);
num_test  = numel(test_image.Files);
num_val   = numel(val_image.Files);

%% Image Preprocessing
%Specify the whitening coefficient 
epsilon = 0.1;

%Training images 
%Read the training images and convert to double
I = readimage(train_image, 1);
I = double(I);

%Preallocate an hxwxcxs array where h is the height, w is the width, c is
%the channel, and s is the number of images
trainimage       = zeros([size(I) 1 num_train], class(I));
norm_train_image = zeros([size(I) 1 num_train], class(I));
std_train_image  = zeros([size(I) 1 num_train], class(I));
zca_train_image  = zeros([size(I) 1 num_train], class(I));

%Feed preprocessed images into the array 
for i = 1: num_train
    trainimage(:, :, i)       = readimage(train_image, i);
    %Mean Normalisation
    norm_train_image(:, :, i) = trainimage(:, :, i) - mean2(trainimage(:, :, i)); 
    %Standardisation
    std_train_image(:, :, i)  = norm_train_image(:, :, i)./ std2(trainimage(:, :, i));
    %ZCA
    trainImage(:, :, i)      = trainimage(:, :, i)/255;
    norm_trainimage(:, :, i) = trainImage(:, :, i) - mean2(trainImage(:, :, i)); 
    c(:, :, i) = cov(norm_trainimage(:, :, i));
    [U(:, :, i), S(:, :, i), V(:, :, i)] = svd(c(:, :, i));
    zca_train_image(:, :, i) = U(:, :, i) .* diag(1/sqrt(diag(S(:, :, i))+epsilon))...
                                .* U(:, :, i).' .* norm_trainimage(:, :, i);   
end

%Validation images 
%Read the validation images and convert to double
J = readimage(val_image, 1);
J = double(J);

%Preallocate an hxwxcxs array where h is the height, w is the width, c is
%the channel, and s is the number of images
valimage       = zeros([size(J) 1 num_val], class(J));
norm_val_image = zeros([size(J) 1 num_val], class(J));
std_val_image  = zeros([size(J) 1 num_val], class(J));
zca_val_image  = zeros([size(J) 1 num_val], class(J));

%Feed preprocessed images into the array 
for j = 1: num_val
    valimage(:, :, j)     = readimage(val_image, j);
    %Mean Normalisation
    norm_val_image(:, :, j) = valimage(:, :, j) - mean2(valimage(:, :, j)); 
    %Standardisation
    std_val_image(:, :, j)  = norm_val_image(:, :, j)./ std2(valimage(:, :, j));
    %ZCA
    valImage(:, :, j)      = valimage(:, :, j)/255;
    norm_valimage(:, :, j) = valImage(:, :, j) - mean2(valImage(:, :, j));
    c1(:, :, j)         = cov(norm_valimage(:, :, j));
    [U1(:, :, j), S1(:, :, j), V1(:, :, j)] = svd(c1(:, :, j));
    zca_val_image(:, :, j)  = U1(:, :, j) .* diag(1/sqrt(diag(S1(:, :, j))+epsilon))...
                              .* U1(:, :, j).' .* norm_valimage(:, :, j);   
end

%Testing images 
%Read the testing images and convert to double
K = readimage(test_image, 1);
K = double(K);

%Preallocate an hxwxcxs array where h is the height, w is the width, c is
%the channel, and s is the number of images
testimage       = zeros([size(K) 1 num_test], class(K));
norm_test_image = zeros([size(K) 1 num_test], class(K));
std_test_image  = zeros([size(K) 1 num_test], class(K));
zca_test_image  = zeros([size(K) 1 num_test], class(K));

%Feed preprocessed images into the array 
for k = 1: num_test
    testimage(:, :, k)       = readimage(test_image, k);
    %Mean Normalisation
    norm_test_image(:, :, k) = testimage(:, :, k) - mean2(testimage(:, :, k)); 
    %Standardisation
    std_test_image(:, :, k)  = norm_test_image(:, :, k)./ std2(testimage(:, :, k));
    %ZCA
    testImage(:, :, k)      = testimage(:, :, k)/255;
    norm_testimage(:, :, k) = testImage(:, :, k) - mean2(testImage(:, :, k));
    c2(:, :, k) = cov(norm_testimage(:, :, k));
    [U2(:, :, k), S2(:, :, k), V2(:, :, k)] = svd(c2(:, :, k));
    zca_test_image(:, :, k) = U2(:, :, k) .* diag(1/sqrt(diag(S2(:, :, k))+epsilon))...
                              .* U2(:, :, k).' .* norm_testimage(:, :, k);   
end

%Training + validation images 
%Read the testing images and convert to double
L = readimage(Train_image, 1);
L = double(L);

%Preallocate an hxwxcxs array where h is the height, w is the width, c is
%the channel, and s is the number of images
Trainimage       = zeros([size(L) 1 Num_train], class(L));
norm_Train_image = zeros([size(L) 1 Num_train], class(L));
std_Train_image  = zeros([size(L) 1 Num_train], class(L));
zca_Train_image  = zeros([size(L) 1 Num_train], class(L));

%Feed preprocessed images into the array 
for l = 1: Num_train
    Trainimage(:, :, l)       = readimage(Train_image, l);
    %Mean Normalisation
    norm_Train_image(:, :, l) = Trainimage(:, :, l) - mean2(Trainimage(:, :, l)); 
    %Standardisation
    std_Train_image(:, :, l)  = norm_Train_image(:, :, l)./ std2(Trainimage(:, :, l));
    %ZCA
    TrainImage(:, :, l)      = Trainimage(:, :, l)/255;
    norm_Trainimage(:, :, l) = TrainImage(:, :, l) - mean2(TrainImage(:, :, l));
    c3(:, :, l) = cov(norm_Trainimage(:, :, l));
    [U3(:, :, l), S3(:, :, l), V3(:, :, l)] = svd(c3(:, :, l));
    zca_Train_image(:, :, l) = U3(:, :, l) .* diag(1/sqrt(diag(S3(:, :, l))+epsilon))...
                               .* U3(:, :, l).' .* norm_Trainimage(:, :, l);   
end
