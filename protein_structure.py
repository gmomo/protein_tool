#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:39:28 2018

@author: soumyadipghosh
"""
#%%
import pandas as pd
import numpy as np
import copy
import math
import sys
from sklearn import linear_model,cross_validation,preprocessing,cluster
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,auc,roc_curve,f1_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold,train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
    
df = pd.read_excel("data_final.xlsx", sheetname=None)
prot_data = df["Sheet1"]
prot_data.drop(prot_data.columns[len(prot_data.columns)-1], axis=1, inplace=True)
prot_data.replace([np.inf, -np.inf], np.nan)
prot_data.dropna(inplace=True)
labels = np.array(prot_data.columns.values,dtype="str")

#%%
prot_data[labels[21:]] = prot_data[labels[21:]].astype(str).replace("-", "0").astype(float)
print (prot_data.dtypes)
uni_protid = prot_data[["UniProt ID"]]
prot_data.drop(["UniProt ID"], axis=1, inplace=True)
#for i in prot_data.index.values:
#    if prot_data["LenDR"][i] < 0.15:
#        prot_data["LenDR"][i] = 0
#    elif prot_data["LenDR"][i] > 0.15 and prot_data["LenDR"][i] < 0.35: 
#        prot_data["LenDR"][i] = 1
#    else:
#        prot_data["LenDR"][i] = 2
        
prot_data.to_csv("prot_data.csv")

y = copy.deepcopy(prot_data["LenDR"])
prot_data.drop(["LenDR"], axis=1, inplace=True)

kf = KFold(n_splits=10,shuffle=True)

auc_base_logistic = []
acc_base_logistic = []

auc_base_svm = []
acc_base_svm = []

#from imblearn.under_sampling import RandomUnderSampler

prot_data = prot_data.as_matrix()
nan_rows = np.isnan(prot_data).any(axis=1)
prot_data = prot_data[~nan_rows,:]
y = y.as_matrix()
y = y[~nan_rows]

X_train_whole, X_test_whole, y_train_whole, y_test_whole = train_test_split(prot_data, y, test_size=0.33,shuffle=True)
#%%
from sklearn.linear_model import ElasticNet,Lasso
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
from sklearn.neural_network import MLPRegressor,BernoulliRBM

whole_mse = []
whole_ev = []

parameters = {'hidden_layer_sizes': [(100,100),(100,50),(100,),(200,100)],
              'alpha':[0.0001, 0.001,0.01,0.1]
              }

mlp_whole = MLPRegressor()
clf = GridSearchCV(mlp_whole, parameters)
clf.fit(X_train_whole,y_train_whole)

dec_val_test=clf.predict(X_test_whole)
whole_mse.append(mean_squared_error(y_test_whole, dec_val_test))
whole_ev.append(explained_variance_score(y_test_whole, dec_val_test))

#sample_dict = {0:16000,1:7000,2:6000}
#
#rus = RandomUnderSampler(sample_dict)
#x_res, y_res = rus.fit_sample(prot_data, y)

#%%
#prot_data = normalize(prot_data,axis=0)
f1_score_list = []
mse_list = []
ev_list = []
mse_lasso_list = []
ev_lasso_list = []

mse_mlp_list = []
ev_mlp_list = []

mlp = MLPRegressor(hidden_layer_sizes=(200,),activation="relu",early_stopping=False)
min_max_scaler = preprocessing.MinMaxScaler()
for train,test in kf.split(prot_data,y):
    x_train = prot_data[train]
    y_train = y[train].ravel()
    
    x_test = prot_data[test]
    y_test = y[test].ravel()
    
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)
    
#    logit = LogisticRegression(multi_class = "multinomial",solver="sag",penalty="l2")
#    
#    logit.fit(x_train,y_train)
#    
#    param_grid = {'C': [10, 100, 1000,10000] }
#    clf = GridSearchCV(linear_model.LogisticRegression(multi_class="multinomial",penalty='l2',solver="sag",class_weight="balanced"), param_grid)
#    clf.fit(x_train,y_train)
    
    regr = ElasticNet(random_state=0)
    regr.fit(x_train, y_train)
    
    mlp.fit(x_train,y_train)
     
    lasso = Lasso(alpha=0.1)
    lasso.fit(x_train,y_train)
    
    dec_val_test=regr.predict(x_test)
    #auc_base_logistic.append(confusion_matrix(y_test,dec_val_test))
    mse_list.append(mean_squared_error(y_test, dec_val_test))
    ev_list.append(explained_variance_score(y_test, dec_val_test) )
    
    dec_val_test=lasso.predict(x_test)
    mse_lasso_list.append(mean_squared_error(y_test, dec_val_test))
    ev_lasso_list.append(explained_variance_score(y_test, dec_val_test))
    
    dec_val_test=mlp.predict(x_test)
    mse_mlp_list.append(mean_squared_error(y_test, dec_val_test))
    ev_mlp_list.append(explained_variance_score(y_test, dec_val_test))
    
    
#%%
import pickle
filename = 'mlp_model.sav'
pickle.dump(mlp, open(filename, 'wb'))
filename_1="minmaxscaler.sav"
pickle.dump(min_max_scaler,open(filename_1,"wb"))
#    pred_val_test=logit.predict(x_test) 
#    acc_base_logistic.append(accuracy_score(y_test,pred_val_test))
#    
#    f1_score_list.append(f1_score(y_test,pred_val_test, average='weighted'))
    
#    rfc = RandomForestClassifier()
#    
#    rfc.fit(x_train,y_train)
#    
##    dec_val_test=logit.decision_function(x_test)
##    auc_base_svm.append(roc_auc_score(y_test,dec_val_test))
#
#    pred_val_test=rfc.predict(x_test) 
#    acc_base_svm.append(accuracy_score(y_test,pred_val_test))
    
    
    


