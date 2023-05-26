# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:24:15 2022

@author: a701192
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# Load data first, I will use german.data-numeric here s that I don't need transfer categorical attributs into integer.
data=pd.read_csv("german.data-numeric", delim_whitespace=True,header=None)
columns=['checking account','duration','credit history','credit amount','saving account','employment','gender and status',
         'residence','property','age','installment plan','existing credits','liable people','telephone','foreign worker',
         'indicator1','indicator2','indicator3','indicator4','indicator5','indicator6','indicator7','indicator8','indicator9','target']
data.columns=columns
#data.isnull().sum() there i no missing value
#For better understanding, change the target column into 0 if class is bad and 1 if class is good 
data['target']=data.target.apply(lambda a: 0 if a==2 else 1)

def age_group2(age):        
    if (age>18 and age<=25):
        return 1    
    else:
        return 0   
    
data['age_cat']=data.age.apply(lambda a: age_group2(a))



data_ageV=data.age_cat.value_counts()
train=pd.DataFrame(columns=data.columns)
test=pd.DataFrame(columns=data.columns)
for age_v in data_ageV.index:
    data_sub=data[data.age_cat==age_v]
    train_sub=data_sub.sample(frac=0.5,random_state=1)
    test_sub=data_sub.drop(train_sub.index)
    train=pd.concat([train,train_sub])
    test=pd.concat([test,test_sub])
    
    
    



train_s=train['age_cat'].values
train_y=train['target'].values
train_x=train.drop(['age_cat','age','target'],axis=1).values 


scaler = StandardScaler()
train_X = scaler.fit_transform(train_x) 
np.savetxt('german age/german_train_x.csv',train_X,delimiter=',')
np.savetxt('german age/german_train_s.csv',train_s,delimiter=',')
np.savetxt('german age/german_train_y.csv',train_y,delimiter=',')

#test
test_s=test['age_cat'].values
test_y=test['target'].values
test_x=test.drop(['age_cat','age','target'],axis=1).values 


scaler2 = StandardScaler()
test_X = scaler2.fit_transform(test_x) 
np.savetxt('german age/german_test_x.csv',test_X,delimiter=',')
np.savetxt('german age/german_test_s.csv',test_s,delimiter=',')
np.savetxt('german age/german_test_y.csv',test_y,delimiter=',')

