# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:38:08 2018

@author: stang040218
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

# Read wine data
data_wine = pd.read_csv("data/wine/winequality-white.csv",sep=';')

# Wine data distribution
dataValue_wine = data_wine.values
features_wine = dataValue_wine[:,0:11]
labels_wine = dataValue_wine[:,-1]

plt.hist(labels_wine,bins = [1,2,3,4,5,6,7,8,9,10]) 
plt.title("histogram") 
plt.show()

# Read credit data
data_credit = pd.read_csv("data/loan/hmeq.csv",sep=',')

# Credit data distribution
dataValue_credit = data_credit.values
features_credit = dataValue_credit[:,1:12]
labels_credit = data_credit['BAD'].astype(str).astype(int)

plt.hist(labels_credit,bins = [0,0.3,0.7,1]) 
plt.title("histogram") 
plt.show()

# Credit data - data cleaning
clean_credit = data_credit

# Create new feature based on whether missing
clean_credit['MISSING_VALUE'] = pd.isnull(data_credit['VALUE'])
clean_credit['MISSING_DEBTINC'] = pd.isnull(data_credit['DEBTINC'])

# Create new value for features if data is missing
clean_credit['REASON'].fillna("NotDisclosed",inplace=True)
clean_credit['JOB'].fillna("NotDisclosed",inplace=True)

# Inputation
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp1 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
clean_credit['MORTDUE'] = imp.fit_transform(clean_credit['MORTDUE'].values.reshape(-1,1))
clean_credit['VALUE'] = imp.fit_transform(clean_credit['VALUE'].values.reshape(-1,1))
clean_credit['YOJ'] = imp.fit_transform(clean_credit['YOJ'].values.reshape(-1,1))
clean_credit['DEROG'] = imp1.fit_transform(clean_credit['DEROG'].values.reshape(-1,1))
clean_credit['DELINQ'] = imp1.fit_transform(clean_credit['DELINQ'].values.reshape(-1,1))
clean_credit['CLAGE'] = imp.fit_transform(clean_credit['CLAGE'].values.reshape(-1,1))
clean_credit['NINQ'] = imp.fit_transform(clean_credit['NINQ'].values.reshape(-1,1))
clean_credit['CLNO'] = imp.fit_transform(clean_credit['CLNO'].values.reshape(-1,1))
clean_credit['DEBTINC'] = imp.fit_transform(clean_credit['DEBTINC'].values.reshape(-1,1))

# Data preprocessing
cleanup_nums = {"REASON": {"DebtCon": 0, "HomeImp": 1, "NotDisclosed": 2},
"JOB": {"Other": 0, "ProfExe": 1, "Office": 2, "Mgr": 3,
"NotDisclosed": 4, "Self": 5, "Sales":6 }}
clean_credit.replace(cleanup_nums, inplace=True)
credit_col = clean_credit.columns
clean_credit[credit_col] = clean_credit[credit_col].apply(pd.to_numeric, errors='coerce')
clean_credit_array=clean_credit.values
credit_y=clean_credit_array[:,0].astype(str).astype(int)
credit_X=clean_credit_array[:,1:15]
# Feature selection
# Wine
feature_scores_wine = mutual_info_classif(features_wine, labels_wine)
print (data_wine.columns)
print (feature_scores_wine)

# Credit
feature_scores_credit = mutual_info_classif(credit_X, credit_y)
print (clean_credit.columns)
print (feature_scores_credit)

wine_selected = ['volatile acidity','residual sugar','chlorides','total sulfur dioxide','density','alcohol','quality']
wine_out=data_wine[wine_selected]
wine_out.to_csv("data/wine/wine_out.csv")

credit_selectedFeatures = ['LOAN','VALUE','DELINQ','CLAGE','DEBTINC','MISSING_DEBTINC']
#scaler = preprocessing.StandardScaler().fit(clean_credit[credit_selectedFeatures])
#credit_out = scaler.transform(clean_credit[credit_selectedFeatures])
#credit_out = np.array(credit_out)
credit_out=clean_credit[credit_selectedFeatures]
out = pd.DataFrame(data = credit_out,columns=credit_selectedFeatures)
out['BAD'] = clean_credit['BAD']
out.to_csv("data/loan/credit_out.csv")

# random sample generation for Credit Dataset
# for Assignment 2 Part I
p = 0.8
sample_Index = np.random.choice(np.arange(credit_out.shape[0]),int(credit_out.shape[0]*p),replace = False)
np.savetxt("data/loan/sample_Index.csv", sample_Index, delimiter=",")
test_Index = list(set(np.arange(features_credit.shape[0]))-set(sample_Index))
test_Index = np.array(test_Index)
np.savetxt("data/loan/test_Index.csv", test_Index, delimiter=",")