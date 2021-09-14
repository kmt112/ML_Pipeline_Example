#!/usr/bin/env python
# coding: utf-8

# In[61]:


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer, mean_absolute_error
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

import xgboost
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[62]:

def gridsearchcv_params_reg(x_train, x_test, y_train, y_test):
    #x_train= pd.read_csv(r'data/x_train.csv')
    #y_train = pd.read_csv(r'data/y_train.csv')
    #x_test = pd.read_csv(r'data/x_test.csv')
    #y_test = pd.read_csv(r'data/y_test.csv')
    #x_train = x_train.iloc[: , 1:]
    #x_test = x_test.iloc[: , 1:]
    #y_train = y_train.iloc[: , 1:]
    #y_test = y_test.iloc[: , 1:]


    # we will be using random forest as a baseline model. Two other models that we will be using to assess the model will be XGBoost and Light GBM

    # ## XGBoost Hyper-Param Tuning
    import xgboost as xgb
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    xgb1 = XGBRegressor()
    params = {
    # Parameters that we are going to tune. values have been tuned previously
        'max_depth': [5], #:[5,6,7],
        'min_child_weight': [6],#[5,6,7],
        'eta':[.05], #[3,1,0.5,0.1,0.05],
        'subsample': [1.0], #[],
        'colsample_bytree': [0.6],
        'n_estimators': [500],
        # Other parameters
        'objective':['reg:squarederror'],
        'eval_metric':['mae']
    }
    #CV is usually set at 3 or 5
    xgb_grid = GridSearchCV(xgb1,
                            params,
                            cv = 2,
                            n_jobs = -1,
                            verbose=False)#turn on verbose if you wanna see it run
    xgb_grid.fit(x_train, y_train)
    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)
    params_xgb_best = xgb_grid.best_params_
    print('XGB_params done')
    # ## LightGBM Hyper-Param Tuning
    
    def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=2, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=False
        )
        fitted_model = gs.fit(X_train_data, y_train_data)
        
        if do_probabilities:
          pred = fitted_model.predict_proba(X_test_data)
        else:
          pred = fitted_model.predict(X_test_data)
        
        return fitted_model, pred
    model = lgb.LGBMRegressor()
    param_grid = {
        'n_estimators': [700],#[700, 1000],
        'colsample_bytree': [0.7],#[0.7,],
        'max_depth': [15], #[15,20],
        'num_leaves': [50], #[50, 100],
        'reg_alpha': [1.1],
        'reg_lambda': [1.3],
        'min_split_gain': [0.3],
        'subsample': [0.9], #[0.8,09],
        'subsample_freq': [20]
        }
    #CV is usually set at 3 or 5    
    model, pred = algorithm_pipeline(x_train, x_test, y_train, y_test, model, 
                                        param_grid, cv=2, scoring_fit='neg_mean_absolute_error')
    print('LGB_params done')    
    print(model.best_score_)
    params_LGBM_best= model.best_params_


    return params_xgb_best, params_LGBM_best


def model_results_reg(params_xgb_best,params_LGBM_best,x_train,x_test,y_train,y_test):
    
    ##XGB
    print('predicting XGB...')
    xgb = XGBRegressor(**params_xgb_best)
    xgb.fit(x_train,y_train)
    y_preds_XGB=xgb.predict(x_test)
    mae_XGB = mean_absolute_error(y_test,y_preds_XGB)
    print('XGB done!')
    
    # Linear Regression
    from sklearn import preprocessing
    normalizer = preprocessing.Normalizer()
    normalized_train_X = normalizer.fit_transform(x_train)
    normalized_train_X
    normalized_test_X = normalizer.transform(x_test)
    normalized_test_X
    regr = LinearRegression()
    regr.fit(normalized_train_X, y_train)
    print('predicting Lin Reg...')
    y_pred_LR = regr.predict(normalized_test_X)
    mae_LR= mean_absolute_error(y_test,y_pred_LR)
    print(mean_absolute_error(y_test,y_pred_LR))
    print('Lin Reg done!')

    #LGBM    
    #params_lgb=model.best_params_
    LGBM = lgb.LGBMRegressor(**params_LGBM_best)
    LGBM.fit(x_train,y_train)
    print('predicting LGBM...')
    y_preds_LGBM = LGBM.predict(x_test)
    mae_LGBM = mean_absolute_error(y_test,y_preds_LGBM)
    print('LGBM done!')

    print(("Linear Regression MAE: {}".format(mae_LR)))
    print(("XGBoost MAE: {}".format(mae_XGB)))
    print(("LGBM MAE: {}".format(mae_LGBM)))
    return mae_XGB,mae_LR,mae_LGBM