%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A Baysian Hyper-parameter Optimisaion is carried out that minimise
% 10-fold stratified cross-validation error, the results will inspire the
% selection of they hyper-parameter adjustment in the training stage in
% SVMs_Training.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Work Space 
close all; clear all; clc

%% Import Scripts of HOG Features Extracted 
run('SVMs_Traning_HOG.m')

%% Begin 10-fold Stratified Cross-Validation
%Specify number of folds 
k = 10;

%Specify cross-validation partition
cvp = cvpartition(Train_label, 'KFold', k);

%Specify class names 
class_name = {'subject01','subject02','subject03','subject04','subject05',...
              'subject06','subject07','subject08','subject09','subject10'};

%For reproducibility           
rng default       

%Train a model to optimise cross-validation error by adjust 
%hyperparameters and use grid search to adjust the hyperparameters further          
Mdl = fitcecoc(Training_features, Train_label, 'OptimizeHyperparameters',...
              'auto', 'HyperparameterOptimizationOptions',...
              struct('AcquisitionFunctionName','expected-improvement-plus'),...
              'ClassNames', class_name, 'ShowPlots', true, 'CVPartition', cvp);          