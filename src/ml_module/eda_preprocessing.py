#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sqlite3 as sql

def eda_preprocessing():
    #database= "AIAP/data/score.db"
    database = "data/score.db"
    connection =sql.connect(database)
    query = '''SELECT * FROM score'''
    df = pd.read_sql_query(query,connection)
    print('extracting data...')
    df.head() #ensure that the data is properly loaded

    df['final_test']=df['final_test'].fillna(df['final_test'].mean())
    df['attendance_rate']=df['attendance_rate'].fillna(df['attendance_rate'].median())
    #check if any null left
    df.isnull().sum()

    from datetime import datetime, timedelta

    df['col'] = pd.to_datetime(df['sleep_time'])
    df['col2'] = pd.to_datetime(df['wake_time'])
    df['datetime_sleep'] = df['col'].dt.strftime('%H:%M')
    df['datetime_wake'] = df['col2'].dt.strftime('%H:%M')
    #(df.fr-df.to).astype('timedelta64[h]')
    df['col_hr'] = df.col.dt.hour
    df['col_min'] = df.col.dt.minute
    df['col2_hr'] = df.col2.dt.hour
    df['col2_min'] = df.col2.dt.minute

    #add 24 to wake hours
    df['col2_hr']=df['col2_hr']+24
    df['total_sleep_hr']= df["col2_hr"] - df["col_hr"]
    # mins
    df['total_sleep_mins']=df["col2_min"] - df["col_min"]

    df=df.drop(['col', 'col2','datetime_sleep','datetime_wake','col_min','col_hr','col2_hr','col2_min','total_sleep_mins'], axis=1)
    df['age']=df['age'].astype(int)
    df = df[df.age > 14]  

    df = df.drop(['index', 'student_id','bag_color','wake_time'], axis=1)
    df["CCA"].replace({"CLUBS": "Clubs", "SPORTS": "Sports", "ARTS":"Arts","NONE":"None"}, inplace=True)
    df["tuition"].replace({"No": "N", "Yes": "Y"}, inplace=True)
    df["sleep_time"].replace({"1:00": "Past_midnight", "3:00": "Past_midnight", "1:30":"Past_midnight","2:00":"Past_midnight","2:30":"Past_midnight","0:30":"Past_midnight","23:30": "Pre_midnight", "21:30": "Pre_midnight", "22:00":"Pre_midnight","21:00":"Pre_midnight","23:00":"Pre_midnight","0:00":"Pre_midnight","22:30":"Pre_midnight"}, inplace=True)
    
    print('done!')

    return df

