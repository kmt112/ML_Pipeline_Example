#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd



from ml_module.eda_preprocessing import eda_preprocessing
from ml_module.data_prep import train_test_data
from ml_module.model_regression import gridsearchcv_params_reg, model_results_reg
from ml_module.model_classification import gridsearchcv_params_class, model_results_class

dataset = eda_preprocessing() #data Processing
#print(dataset)
x_train, x_test, y_train, y_test = train_test_data(dataset) # train test split
#print(y_train)

#run if you need to optimize hyper parameters
params_XGB_reg, params_LGBM_reg = gridsearchcv_params_reg(x_train, x_test, y_train, y_test)

#MAE for regression
mae_XGB_reg, mae_LR_reg, mae_LGBM_reg= model_results_reg(params_XGB_reg,params_LGBM_reg,x_train,x_test,y_train,y_test)

#run if you need to optimize hyper parameters
params_LR_class, params_RF_class, params_XGB_class = gridsearchcv_params_class(x_train, x_test, y_train, y_test)

#MAE for classification

mae_LR_class, mae_RF_class, mae_XGB_class = model_results_class(params_LR_class,params_RF_class,params_XGB_class,x_train,x_test,y_train,y_test)

#Printing results onto df
results_df = pd.DataFrame(np.array([["Linear regression", mae_LR_reg], ["XGB_regressor", mae_XGB_reg], ["LGBM_regressor", mae_LGBM_reg],["Logistic regression", mae_LR_class *10 ],["Random forest Classifier", mae_RF_class *10],["XGB Classifier", mae_XGB_class *10]]),
                   #columns=['Model', 'MAE'])

#results_df = pd.DataFrame(np.array([["Logistic regression", mae_LR_class *10 ],["Random forest Classifier", mae_RF_class *10],["XGB Classifier", mae_XGB_class *10]]),
                   columns=['Model', 'MAE'])

results_df.to_csv(r'results/MAE.csv', index=False, header=False)