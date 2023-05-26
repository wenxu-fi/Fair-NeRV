# -*- coding: utf-8 -*-
"""
This file preprocess adult data, choose some features and 1000 points, split into train and test data half to half and
 saved in fold "adult gender" 
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

adult= pd.read_csv("adult-clean.csv")
adult.isnull().sum()

adult['gender']=adult.gender.apply(lambda a:0 if a=='Male' else 1)
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship','native-country']
for col in category_col:
    b, c = np.unique(adult[col], return_inverse=True)
    adult[col] = c


data=adult.copy()
data_n=data.shape[0] 
sens_VC=data.gender.value_counts()


train=pd.DataFrame(columns=data.columns)
test=pd.DataFrame(columns=data.columns)
for s in sens_VC.index:
    data_s=data[data.gender==s]
    sub_n=round(1000*(sens_VC[s]/data_n))
    data_sub=data_s.sample(n=sub_n,random_state=1)
    train_n=round(sub_n/2)
    train_sub=data_sub.iloc[:train_n,:]
    test_sub=data_sub.iloc[train_n:,:]
    train=pd.concat([train,train_sub])
    test=pd.concat([test,test_sub])
    
train_s=train['gender'].values
train_y=train['Class-label'].values
train_x=train.drop(['gender','Class-label'],axis=1).values 
train_xy=train.drop(['gender'],axis=1).values 

scaler = StandardScaler()
train_XY = scaler.fit_transform(train_xy) 
train_X=scaler.fit_transform(train_x)
np.savetxt('adult gender/adult_train_x.csv',train_X,delimiter=',')

np.savetxt('adult gender/adult_train_xy.csv',train_XY,delimiter=',')
np.savetxt('adult gender/adult_train_s.csv',train_s,delimiter=',')
np.savetxt('adult gender/adult_train_y.csv',train_y,delimiter=',')

#testXY
test_s=test['gender'].values
test_y=test['Class-label'].values
test_x=test.drop(['gender','Class-label'],axis=1).values 
test_xy=test.drop(['gender'],axis=1).values 


scaler2 = StandardScaler()
test_XY = scaler2.fit_transform(test_xy) 
test_X = scaler2.fit_transform(test_x) 

np.savetxt('adult gender/adult_test_x.csv',test_X,delimiter=',')
np.savetxt('adult gender/adult_test_xy.csv',test_XY,delimiter=',')
np.savetxt('adult gender/adult_test_s.csv',test_s,delimiter=',')
np.savetxt('adult gender/adult_test_y.csv',test_y,delimiter=',')
    