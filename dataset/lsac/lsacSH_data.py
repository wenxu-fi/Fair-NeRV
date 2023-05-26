# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:25:32 2022

@author: a701192
"""
import pandas as pd
import numpy as np

from scipy.io import loadmat
from sklearn.metrics import pairwise_distances
#from sklearn.manifold import _utils
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
import random
from sklearn.manifold import TSNE
#import csv
import pickle
from sklearn.decomposition import PCA
from IPython import display

lsac_data=pd.read_csv('lsac_x.csv',names=range(13),header=0) #
lsac_sens=pd.read_csv('lsac_s.csv',names=['race','age'],header=0)
lsac_y=pd.read_csv('lsac_y.csv',header=0)
lsac_s=lsac_sens['race']
data=lsac_data.assign(race=lsac_s)
data['y']=lsac_y
#lsac_sens['race'].unique()  # 0,1
#lsac_sens['age'].unique() # 51



data_n=data.shape[0] 
sens_VC=data.race.value_counts()
'''
data_1=data[data.race==1]
sub_1=round(1000*(sens_VC[1]/data_n))
data_sub=data_1.sample(n=sub_1,random_state=1)
train_sub=data_sub.iloc[:round(sub_1/2),:]
'''

train=pd.DataFrame(columns=data.columns)
test=pd.DataFrame(columns=data.columns)
for s in sens_VC.index:
    data_s=data[data.race==s]
    sub_n=round(1000*(sens_VC[s]/data_n))
    data_sub=data_s.sample(n=sub_n,random_state=1)
    train_n=round(sub_n/2)
    train_sub=data_sub.iloc[:train_n,:]
    test_sub=data_sub.iloc[train_n:,:]
    train=pd.concat([train,train_sub])
    test=pd.concat([test,test_sub])
    
#train['race'].sum()/500
        
train_s=train['race'].values
train_y=train['y'].values
train_x=train.drop(['race','y'],axis=1).values 


scaler = StandardScaler()
train_X = scaler.fit_transform(train_x) 
np.savetxt('lsacSH/lsac_train_x.csv',train_X,delimiter=',')
np.savetxt('lsacSH/lsac_train_s.csv',train_s,delimiter=',')
np.savetxt('lsacSH/lsac_train_y.csv',train_y,delimiter=',')

#test
test_s=test['race'].values
test_y=test['y'].values
test_x=test.drop(['race','y'],axis=1).values 


scaler2 = StandardScaler()
test_X = scaler2.fit_transform(test_x) 
np.savetxt('lsacSH/lsac_test_x.csv',test_X,delimiter=',')
np.savetxt('lsacSH/lsac_test_s.csv',test_s,delimiter=',')
np.savetxt('lsacSH/lsac_test_y.csv',test_y,delimiter=',')
    