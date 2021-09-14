#!/usr/bin/env python
# coding: utf-8

# In[106]:


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer, mean_absolute_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import xgboost
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def gridsearchcv_params_class(x_train, x_test, y_train, y_test):
    #encoding scores
    grades = [1,2,3,4,5,6,7,8,9,10]
    y_train=pd.DataFrame(y_train)
    y_train.columns=['final_test']
    y_test=pd.DataFrame(y_test)
    y_test.columns=['final_test']
    #print(y_train) #ensures it works
    conditions= [
        (y_train['final_test'] <= 10),
        (y_train['final_test'] > 10) & (y_train['final_test'] <= 20),
        (y_train['final_test'] > 20) & (y_train['final_test'] <= 30),
        (y_train['final_test'] > 30) & (y_train['final_test'] <= 40),
        (y_train['final_test'] > 40) & (y_train['final_test'] <= 50),
        (y_train['final_test'] > 50) & (y_train['final_test'] <= 60),
        (y_train['final_test'] > 60) & (y_train['final_test'] <= 70),
        (y_train['final_test'] > 70) & (y_train['final_test'] <= 80),
        (y_train['final_test'] > 80) & (y_train['final_test'] <= 90),
        (y_train['final_test'] > 90) & (y_train['final_test'] <= 100)
        ]
    
    conditions_test= [
        (y_test['final_test'] <= 10),
        (y_test['final_test'] > 10) & (y_test['final_test'] <= 20),
        (y_test['final_test'] > 20) & (y_test['final_test'] <= 30),
        (y_test['final_test'] > 30) & (y_test['final_test'] <= 40),
        (y_test['final_test'] > 40) & (y_test['final_test'] <= 50),
        (y_test['final_test'] > 50) & (y_test['final_test'] <= 60),
        (y_test['final_test'] > 60) & (y_test['final_test'] <= 70),
        (y_test['final_test'] > 70) & (y_test['final_test'] <= 80),
        (y_test['final_test'] > 80) & (y_test['final_test'] <= 90),
        (y_test['final_test'] > 90) & (y_test['final_test'] <= 100)
        ]
    
    y_train['new_score']=np.select(conditions,grades)
    y_test['new_score']=np.select(conditions_test,grades)
    y_train = y_train.iloc[: , 1:]
    y_test = y_test.iloc[: , 1:]
    
    print('Logistic_regression hyperparams')
    model = LogisticRegression()
    solvers = ['liblinear']
    penalty = ['l1']
    max_iter=[10000]
    c_values = [100]
    
    
    from sklearn import preprocessing
    
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(x_train)
    normalized_test_X = normalizer.transform(x_test)
    
    from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
    grid = dict(solver=solvers,penalty=penalty,C=c_values,max_iter=max_iter)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error',error_score=0)
    grid_result = grid_search.fit(normalized_train_X, y_train.values.ravel())
    
    params_LR=grid_result.best_params_
    print('log_reg completed!')
    # ## Random Forest Classifier


    from sklearn.model_selection import GridSearchCV
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80],#[80, 90, 100, 110],
        'max_features': [2],
        'min_samples_leaf': [4],
        'min_samples_split': [10],
        'n_estimators': [1000]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 2, n_jobs = -1, verbose = False)
    
    print('rf starting')
    grid_search.fit(normalized_train_X, y_train.values.ravel())
    grid_search.best_params_
    params_RF = grid_search.best_params_
    print('rf completed')
    
    xgb1 = XGBClassifier()
    params = {
        # Parameters that we are going to tune.
        'max_depth': [5],
        'min_child_weight': [6],
        'eta':[.05 ],
        'subsample': [1.0],
        'colsample_bytree': [0.6],
        'n_estimators': [125],
        # Other parameters
        'objective':['reg:squarederror'],
        'eval_metric':['mae']
    }
    

    xgb_grid = GridSearchCV(xgb1,
                            params,
                            cv = 2,
                            n_jobs = -1,
                            verbose=1)
    print('XGB starting')
    xgb_grid.fit(x_train, y_train.values.ravel())
    print('XGB completed')
    print(xgb_grid.best_params_)
    params_xgb = xgb_grid.best_params_

    return params_LR, params_RF, params_xgb
    
def model_results_class (params_LR,params_RF,params_XGB,x_train,x_test,y_train,y_test):
    #LR MAE
    from sklearn.model_selection import RepeatedStratifiedKFold
    
    grades = [1,2,3,4,5,6,7,8,9,10]
    y_train=pd.DataFrame(y_train)
    y_train.columns=['final_test']
    y_test=pd.DataFrame(y_test)
    y_test.columns=['final_test']
    #y_test.columns=['final_test']
    print(y_train)
    conditions= [
        (y_train['final_test'] <= 10),
        (y_train['final_test'] > 10) & (y_train['final_test'] <= 20),
        (y_train['final_test'] > 20) & (y_train['final_test'] <= 30),
        (y_train['final_test'] > 30) & (y_train['final_test'] <= 40),
        (y_train['final_test'] > 40) & (y_train['final_test'] <= 50),
        (y_train['final_test'] > 50) & (y_train['final_test'] <= 60),
        (y_train['final_test'] > 60) & (y_train['final_test'] <= 70),
        (y_train['final_test'] > 70) & (y_train['final_test'] <= 80),
        (y_train['final_test'] > 80) & (y_train['final_test'] <= 90),
        (y_train['final_test'] > 90) & (y_train['final_test'] <= 100)
        ]
    
    conditions_test= [
        (y_test['final_test'] <= 10),
        (y_test['final_test'] > 10) & (y_test['final_test'] <= 20),
        (y_test['final_test'] > 20) & (y_test['final_test'] <= 30),
        (y_test['final_test'] > 30) & (y_test['final_test'] <= 40),
        (y_test['final_test'] > 40) & (y_test['final_test'] <= 50),
        (y_test['final_test'] > 50) & (y_test['final_test'] <= 60),
        (y_test['final_test'] > 60) & (y_test['final_test'] <= 70),
        (y_test['final_test'] > 70) & (y_test['final_test'] <= 80),
        (y_test['final_test'] > 80) & (y_test['final_test'] <= 90),
        (y_test['final_test'] > 90) & (y_test['final_test'] <= 100)
        ]
    
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error , make_scorer, mean_absolute_error
    
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(x_train)
    normalized_test_X = normalizer.transform(x_test)
    
    y_train['new_score']=np.select(conditions,grades)
    y_test['new_score']=np.select(conditions_test,grades)
    y_train = y_train.iloc[: , 1:]
    y_test = y_test.iloc[: , 1:]
    #print(y_test) check that code is working
    #print(y_train) check that code is working
    regressor = LogisticRegression(**params_LR)
    regressor.fit(normalized_train_X, y_train.values.ravel())
    print('predicting Log Reg...')
    y_preds =regressor.predict(normalized_test_X)
    
    LogReg_MAE = mean_absolute_error(y_preds,y_test)
    print('Log Reg done!')
    #RF MAE
    #best_grid = grid_search.best_estimator_
    rf = RandomForestClassifier(**params_RF)
    rf.fit(normalized_train_X, y_train.values.ravel())
    print('predicting Random forest...')
    y_preds_rf = rf.predict(normalized_test_X)
    RF_MAE = mean_absolute_error(y_preds_rf,y_test)
    print('Random Forest done!')
    
    #XGB MAE
    xgb = XGBClassifier(**params_XGB)
    xgb.fit(x_train,y_train)
    print('predicting XGBclassifier...')
    y_preds_XGB=xgb.predict(x_test)
    XGBC_MAE = mean_absolute_error(y_test,y_preds_XGB)
    print('XGB forest done!')
    print(("Logistic Regression MAE: {}".format(LogReg_MAE)))
    print(("XGBoost MAE: {}".format(XGBC_MAE)))
    print(("Random Forest MAE: {}".format(RF_MAE)))
    return LogReg_MAE,RF_MAE,XGBC_MAE