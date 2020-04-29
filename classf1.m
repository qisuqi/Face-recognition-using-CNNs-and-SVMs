%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is provided by Mathsworks to obtain the mis-classification
% during cross-validation, it is used in the testing stage of SVMs. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function pred = classf1(Training_features, Train_label, testing_features)

mdl  = fitcecoc(Training_features,Train_label);
pred = predict(mdl,testing_features);

end