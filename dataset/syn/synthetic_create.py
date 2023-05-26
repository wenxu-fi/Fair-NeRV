# -*- coding: utf-8 -*-
"""
This file create synthetic data 


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import scipy

#%% functions to simulate multivariate  gaussian mixture  data

def corr2cov(correlations, stdev):
    """Covariance matrix from correlation & standard deviations"""
    d = np.diag(stdev)
    return d @ correlations @ d



def multi_GMM(n_comp, means,stdevs,corrs,mult,N=500,):
    assert n_comp == len(means), "The length of the list of mean values does not match number of Gaussian components"
    assert n_comp == len(stdevs), "The length of the list of sigma values does not match number of Gaussian components"
    assert n_comp == len(mult), "The length of the list of multiplier values does not match number of Gaussian components"
    n=means.shape[1]
    rand_samples = np.zeros((N,n))
    mult=mult/np.sum(mult)
    covs=[]
    for i in range(stdevs.shape[0]):
        stdev=stdevs[i,:]
        corr=corrs[i,:]
        covs.append(corr2cov(corr,stdev))
    comp_id=np.zeros(N)    
    for i in range(N):
        j=random.choices(range(n_comp),mult)[0]
        comp_id[i]=j
        rand_samples[i]= np.random.multivariate_normal(mean=means[j,:],cov=covs[j],size=1)
    return rand_samples,comp_id.astype(int)

def logisticF(x):
    f=1/(1+np.exp(-x))
    random_nums=scipy.stats.uniform.rvs(size=x.shape)
    #f=np.where(f>0.5,1,0)
    f=np.where(f>random_nums,1,0)
    return f

def fourclass(x):
    if x<0.25:
        return 0
    elif x<0.5:
        return 1
    elif x<0.75:
        return 2
    else:
        return 3
def logisticF2(v):
    f=1/(1+np.exp(-v))
    vfun=np.vectorize(fourclass)
   # random_nums=scipy.stats.uniform.rvs(size=x.shape)
    #f=np.where(f>0.5,1,0)
    res=vfun(f)
    return res

#%% creat synthetic data
n_comp1=3
means1=np.array([[-1,0,1],[4,5.5,7],[-10,-8,-6]])
stdevs1=np.array([[2,1,1],[1,1,1],[1,1,1]])
corrs1=np.array([[[1,-0.4,0.8],[-0.4,1,0],[0.8,0,1]],[[1,0.2,-0.1],[0.2,1,-0.7],[-0.1,-0.7,1]],[[1,0.5,0.3],[0.5,0.3,-0.4],[0.3,-0.4,0.4]]])
mult=np.array([1,1,1])
sample1,comp1=multi_GMM(n_comp1,means1,stdevs1,corrs1,mult,1000)
plt.scatter(sample1[:,1],sample1[:,2])

n_comp3=3
means3=np.array([[8,8],[0,2],[-9,-7]])
stdevs3=np.array([[1,1],[2,1],[1,1]])
corrs3=np.array([[[1,0.7],[0.7,1]],[[1,-0.4],[-0.4,1]],[[1,0],[0,1]]])
mult3=np.array([1,1,1])
sample3,comp3=multi_GMM(n_comp3,means3,stdevs3,corrs3,mult3,1000)
plt.scatter(sample3[:,0],sample3[:,1])

features=np.hstack((sample1,sample3))
data=pd.DataFrame(features)
data['sens']=comp1
data['cluster']=comp3
data_n=data.shape[0]
sens_VC=data.sens.value_counts()
#Split 1000 points into train and test data half to half

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

train_s=train['sens'].values
train_cluster=train['cluster'].values
train_x=train.drop(['sens','cluster'],axis=1).values 


scaler = StandardScaler()
train_X = scaler.fit_transform(train_x) 
np.savetxt('syn cluster/syn_rain_x.csv',train_X,delimiter=',')
np.savetxt('syn cluster/syn_train_s.csv',train_s,delimiter=',')
np.savetxt('syn cluster/syn_train_cluster.csv',train_cluster,delimiter=',')

#test
test_s=test['sens'].values
test_cluster=test['cluster'].values
test_x=test.drop(['sens','cluster'],axis=1).values 

plt.scatter(test_x [:,3],test_x[:,4])

scaler2 = StandardScaler()
test_X = scaler2.fit_transform(test_x) 
np.savetxt('syn cluster/syn_test_x.csv',test_X,delimiter=',')
np.savetxt('syn cluster/syn_test_s.csv',test_s,delimiter=',')
np.savetxt('syn cluster/syn_test_cluster.csv',test_cluster,delimiter=',')




