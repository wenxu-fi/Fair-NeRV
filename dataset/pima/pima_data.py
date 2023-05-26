# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:08:26 2022

@author: a701192
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

pima = pd.read_csv("diabetes.csv")
pima.isnull().sum()
pima=pima[pima['BMI']!=0]

def weight(x):
    if x<18.5:
        return 'Underweight'
    elif x<25:
        return 'Normal'
    elif x<30:
        return 'Overweight'
    else:
        return 'Obese'
def weightN(x):
    if x<18.5:
        return 0
    elif x<25:
        return 1
    elif x<30:
        return 2
    else:
        return 3      

def weightN(x):
    if x<25:
        return 0
    elif x<30:
        return 1
    else:
        return 2 
    
   
pima['weightN']=pima['BMI'].apply(weightN)  
pima.hist(bins=50, figsize=(20, 15))
plt.show()


data=pima.copy()
data.rename(columns={'Outcome':'y','weightN':'sens'},inplace=True)
data_n=data.shape[0] 
sens_VC=data.sens.value_counts()
'''
data_1=data[data.race==1]
sub_1=round(1000*(sens_VC[1]/data_n))
data_sub=data_1.sample(n=sub_1,random_state=1)
train_sub=data_sub.iloc[:round(sub_1/2),:]
'''

train=pd.DataFrame(columns=data.columns)
test=pd.DataFrame(columns=data.columns)
for s in sens_VC.index:
    data_s=data[data.sens==s]
    sub_n=min(round(1000*(sens_VC[s]/data_n)),sens_VC[s])
    data_sub=data_s.sample(n=sub_n,random_state=1)
    train_n=round(sub_n/2)
    train_sub=data_sub.iloc[:train_n,:]
    test_sub=data_sub.iloc[train_n:,:]
    train=pd.concat([train,train_sub])
    test=pd.concat([test,test_sub])
    
#train['race'].sum()/500
        
train_s=train['sens'].values
train_y=train['y'].values
train_x=train.drop(['BMI','sens','y'],axis=1).values 
train_xy=train.drop(['BMI','sens'],axis=1).values 


scaler = StandardScaler()
train_X = scaler.fit_transform(train_x) 
np.savetxt('pima bmi/pima_train_x.csv',train_X,delimiter=',')
np.savetxt('pima bmi/pima_train_s.csv',train_s,delimiter=',')
np.savetxt('pima bmi/pima_train_y.csv',train_y,delimiter=',')

#test
test_s=test['sens'].values
test_y=test['y'].values
test_x=test.drop(['BMI','sens','y'],axis=1).values 


scaler2 = StandardScaler()
test_X = scaler2.fit_transform(test_x) 
np.savetxt('pima bmi/pima_test_x.csv',test_X,delimiter=',')
np.savetxt('pima bmi/pima_test_s.csv',test_s,delimiter=',')
np.savetxt('pima bmi/pima_test_y.csv',test_y,delimiter=',')


#train_XY
train_s=train['sens'].values
train_y=train['y'].values
#train_x=train.drop(['BMI','sens','y'],axis=1).values 
train_xy=train.drop(['BMI','sens'],axis=1).values 

scaler = StandardScaler()
train_XY = scaler.fit_transform(train_xy) 
np.savetxt('pima bmi/pima_train_xy.csv',train_XY,delimiter=',')
np.savetxt('pima bmi/pima_train_s.csv',train_s,delimiter=',')
np.savetxt('pima bmi/pima_train_y.csv',train_y,delimiter=',')

#testXY
test_s=test['sens'].values
test_y=test['y'].values
#test_x=test.drop(['BMI','sens','y'],axis=1).values 
test_xy=test.drop(['BMI','sens'],axis=1).values 


scaler2 = StandardScaler()
test_XY = scaler2.fit_transform(test_xy) 
np.savetxt('pima bmi/pima_test_xy.csv',test_XY,delimiter=',')
np.savetxt('pima bmi/pima_test_s.csv',test_s,delimiter=',')
np.savetxt('pima bmi/pima_test_y.csv',test_y,delimiter=',')
