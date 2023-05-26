# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:33:01 2022

@author: a701192
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
import random


cc = pd.read_csv("communities_crime.csv") #datasurvey
'''
cc.isnull().sum()
cc=cc.drop
cc.columns
cc_nameC=cc.communityname.value_counts()
sprinfield=cc[cc.communityname=='Springfieldcity']
'''
category_col=['state','communityname']
for col in category_col:
        b, c = np.unique(cc[col], return_inverse=True)
        cc[col] = c
 
selected_columns=['racepctblack','pctWInvInc','pctWPubAsst','NumUnderPov','PctPopUnderPov',
          'PctUnemployed','MalePctDivorce','FemalePctDiv','TotalPctDiv','PersPerFam',
          'PctKids2Par','PctYoungKids2Par','PctTeen2Par','NumIlleg','PctIlleg',
          'PctPersOwnOccup','HousVacant','PctHousOwnOcc','PctVacantBoarded','NumInShelters',
          'NumStreet','ViolentCrimesPerPop','Black','class']

cc=cc[selected_columns]
        
data=cc.copy()
data_n=data.shape[0] 
sens_VC=data.Black.value_counts()
'''
data_1=data[data.race==1]
sub_1=round(1000*(sens_VC[1]/data_n))
data_sub=data_1.sample(n=sub_1,random_state=1)
train_sub=data_sub.iloc[:round(sub_1/2),:]
'''

train=pd.DataFrame(columns=data.columns)
test=pd.DataFrame(columns=data.columns)
for s in sens_VC.index:
    data_s=data[data.Black==s]
    sub_n=round(1000*(sens_VC[s]/data_n))
    data_sub=data_s.sample(n=sub_n,random_state=1)
    train_n=round(sub_n/2)
    train_sub=data_sub.iloc[:train_n,:]
    test_sub=data_sub.iloc[train_n:,:]
    train=pd.concat([train,train_sub])
    test=pd.concat([test,test_sub])
    
#train['race'].sum()/500
        
train_s=train['Black'].values
train_y=train['class'].values
train_x=train.drop(['racepctblack','Black','class'],axis=1).values 


scaler = StandardScaler()
train_X = scaler.fit_transform(train_x) 
np.savetxt('cc black/cc_train_x.csv',train_X,delimiter=',')
np.savetxt('cc black/cc_train_s.csv',train_s,delimiter=',')
np.savetxt('cc black/cc_train_y.csv',train_y,delimiter=',')

#test
test_s=test['Black'].values
test_y=test['class'].values
test_x=test.drop(['racepctblack','Black','class'],axis=1).values 


scaler2 = StandardScaler()
test_X = scaler2.fit_transform(test_x) 
np.savetxt('cc black/cc_test_x.csv',test_X,delimiter=',')
np.savetxt('cc black/cc_test_s.csv',test_s,delimiter=',')
np.savetxt('cc black/cc_test_y.csv',test_y,delimiter=',')
