#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Method1: Fitting objective functions to MOS using regression
to get better performance than single objective function

Method2: Fitting feature metrics to MOS using  regression

Method3: Fitting images to MOS directly using CNN

Created on Wed Mar  7 14:54:50 2018

@author: bingxinhou

 1      FSIMc           0.851                1      FSIMc           0.667            
 2      PSNR-HA         0.819                2      PSNR-HA         0.643 
 3      PSNR-HMA        0.813                3      PSNR-HMA        0.632            
 4      FSIM            0.801                4      FSIM            0.630            
 5      MSSIM           0.787                5      MSSIM           0.608            
 6      PSNRc           0.687                6      VSNR            0.508            
 7      VSNR            0.681                7      PSNR-HVS        0.508            
 8      PSNR-HVS        0.654                8      PSNRc           0.496            
 9      PSNR            0.640                9      PSNR-HVS-M      0.482            
 10     SSIM            0.637                10     PSNR            0.470            
 11     NQM             0.635                11     NQM             0.466            
 12     PSNR-HVS-M      0.625                12     SSIM            0.464            
 13     VIFP            0.608                13     VIFP            0.457            
 14     WSNR            0.580                14     WSNR            0.446 
''' 


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from scipy.stats import spearmanr,kendalltau
from numpy import array

 # import TID2013
 
file="/Users/bingxinhou/Documents/SCU/338VideoComptression/TID2013/metrics_values/"
dict = { 1  :    'FSIMc.txt'     ,  
         2  :   'PSNRHA.txt'    ,
         3  :    "PSNRHMA.txt" ,                              
         4  :    "FSIM.txt"      ,                             
         5  :    "MSSIM.txt"    ,                             
         6  :    "PSNRc.txt"     ,                               
         7  :    "VSNR.txt"      ,                             
         8  :   "PSNRHVS.txt"    ,                             
         9  :  "PSNR.txt"       ,                          
         10 :  "SSIM.txt"        ,                        
         11 :  "NQM.txt"         ,                          
         12 :  "PSNRHVSM.txt"  ,                            
         13 :  "VIFP.txt"        ,                            
         14 :  "WSNR.txt"}

X=[]
for n in range(1,15):
 X.append(np.loadtxt(file+dict[n]));
X=array(X).T

np.savetxt("FR_IQA_Scores.txt", X, delimiter=" ")

file_Y="/Users/bingxinhou/Documents/SCU/338VideoComptression/TID2013/MOS.txt"
Y = np.loadtxt(file_Y);



X_select = X[:,1:7];
Y_select = Y;


 # machine learning
X_train, X_test, Y_train, Y_test = train_test_split(
        X_select, Y_select, test_size=0.33, random_state=42)

classifiers = [
        svm.SVR(),
        linear_model.LinearRegression(),
        neighbors.KNeighborsRegressor(5, weights='uniform')]


for item in classifiers: 
   
    clf = item
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_select)
#    print(mean_squared_error(Y_test,Y_pred))
#    print(cross_val_score(clf,X_train,Y_train,cv=4))
    
 # calculate covariance
 # PLCC
 
 # SRCC
 # KRCC
 # R2
 # MSE


#rho_1, pval_1 = pearsonr(Y_pred,Y)
rho_2, pval_2 = spearmanr(Y_pred,Y)
rho_3, pval_3 = kendalltau(Y_pred,Y)
print('Method1: '+str([rho_2,rho_3]))


#for n in range(9):
for n in range(0,14):
 #rho1, pval1 = pearsonr(X[:,n],Y)
 rho2, pval2 = spearmanr(X[:,n],Y)
 rho3, pval3 = kendalltau(X[:,n],Y)
 print(dict[n+1]+str([rho2,rho3]))

 

#pearsonr(X_test, y_test)
#for kernel in ('linear', 'poly', 'rbf'):
#    clf = svm.SVC(kernel=kernel, gamma=2)
#    clf.fit(X_train, Y_train)
#    
#Y_pred = clf.predict(X_test)




#kf = KFold(n_splits=2)
#kf.get_n_splits(X)

#KFold(n_splits=2, random_state=None, shuffle=False)
#for train_index, test_index in kf.split(X):
#...    X_train, X_test = X[train_index], X[test_index]
#...    y_train, y_test = y[train_index], y[test_index]

 # define your own score
#pearsonr(X_test, y_test)


