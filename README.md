A Comparative Study in Face Recognition with CNNs and SVMs 
Dataset used for this study is MIT-CBCL Face Recognition Database, it can found here: http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html
------------------------------------------------------------------------------------------
A brief description of each Matlab script is as follows:

CNNs

CNNs_Data_Preparation.m - All necessary data preparation such as importing images in the database to the Image Datastore, resize the images, and split into training, validation, and testing data. Image preprocessing is also conducted. 

CNNs_Baseline.m - A baseline model for CNNs using raw input images, trained on training+validation data and tested on the unseen testing data. 

CNNs_Training_Architecture.m - A grid search is carried out to obtain the best performing CNNs architecture using raw input images. 

CNNs_Training_Parameters.m - A grid search is carried out to adjust the hyper-parameters of CNNs using the best performing architecture obtained from CNNs_Training_Architecture.m, trained on training data and tested on the validation data. 

CNNs_Validating.m - A confusion matrix and ROC curve is plotted of the best performing model obtained from CNNs_Training_Parameters.m.

CNNs_Testing.m - Finally, the best performing model is trained on training+validation data, and tested on the unseen testing data. 

SVMs

SVMs_Data_Preparation.m -  All necessary data preparation such as importing images in the database to the Image Datastore, resize the images, and split into training, validation, and testing data. 

SVMs_HOG_Features.m - HOG Features are extracted from raw input images, and image preprocessing is then conducted. A simple SVMs classifier is also used to train and test each image preprocessing technique, this will be used as the SVMs baseline model. 

SVMs_Opt.m - A Bayesian Optimisation is carried out that minimises 10-fold stratified cross-validation error, and the results will inspire the selection of hyper-parameter adjustment in the training stage. 

SVMs_Training.m - A grid search is carried out to adjust the hyper-parameters of SVMs using HOG features extracted training and validation data and a 10-fold stratified cross-validation. 

SVMs_Validating.m - A confusion matrix and ROC curve is plotted of the best performing model obtained from SVMs_Training.m.

SVMs_Testing_Training.m - The best performing SVMs model is trained with HOG features extracted training and validation data using a 10-fold stratified cross-validation. 

SVMs_Final_20200413_2338.mat - The best performing SVMs model is saved here. Due to the size of this file, it is uploaded to Google drive: https://drive.google.com/drive/folders/1KWmcCt2bc_g5JzGB-Ob5_xPZn76ba_AT

SVMs_Testing_Testing.m - Using saved best performing SVMs model in SVMs_Final_20200413_2338.mat and test on the unseen testing data. 

Functions 

classf.m - This function is provided by Mathworks to obtain the 10-fold stratified cross-validation error, it is used in the training stage of SVMs in SVMS_Training.m. 

classf1.m - This function is provided by Mathworks to obtain the 10-fold stratified cross-validation error, it is used in the testing stage of SVMs in SVMS_Testing.m. 

stopIfAccuracyNotImproving.m - This function is provided by Mathworks and act as an early stopping trigger for training CNNs. It is used in both training and testing stage of CNNs in CNNs_Training_Parameters.m and CNNs_Testing.m. 
