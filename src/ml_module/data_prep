#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

def train_test_data(df):
     
    y= df.final_test
    print('loading train test split...')
    cat_features = ['direct_admission','CCA','learning_style','gender','tuition','sleep_time','mode_of_transport']
    ord_features = ['CCA','sleep_time','number_of_siblings']
    num_features = ['hours_per_week','n_male','n_female','age','attendance_rate']
    
    cat_feat_dum = pd.get_dummies(df[cat_features],drop_first=True)
    
    
    encoder = TargetEncoder()
    ord_feat_dum = encoder.fit_transform(df[ord_features], df['final_test'])
    num_feat_dum = df[num_features]
    
    df_prep2 = pd.concat([ord_feat_dum,cat_feat_dum,num_feat_dum], axis=1)
    
    
    df_prep = pd.concat([y,df_prep2],axis=1)
    
    
    
    y= df_prep.final_test
    x = df_prep.loc[:, df_prep.columns != 'final_test']
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.8, random_state =  90)
    
    
    print('completed!')
    #x_train.to_csv(r'data/x_train.csv')
    #x_test.to_csv(r'data/x_test.csv')
    #y_train.to_csv(r'data/y_train.csv')
    #y_test.to_csv(r'data/y_test.csv')
    return x_train, x_test, y_train, y_test



