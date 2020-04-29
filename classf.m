%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is provided by Mathsworks to obtain the mis-classification
% during cross-validation, it is used in the training stage of SVMs. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function pred = classf(training_features, train_label, validation_features)

mdl = fitcecoc(training_features,train_label);
pred = predict(mdl,validation_features);
end
