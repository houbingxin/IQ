#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:59:52 2018

@author: bingxinhou
"""#!/usr/bin/env python3
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
import sys
import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import math 
from brisque_features import compute_features


    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import spearmanr,pearsonr,kendalltau
from scipy import ndimage
from numpy import array
from brisque_features import compute_features

    
 # import TID2013
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

file="/Users/bingxinhou/Documents/SCU/* 338VideoComptression/TID2013/distorted_images/"
from PIL import Image
import glob

## save the features to txt file and read it for conveinience
#feat_list = []
#for filename in glob.glob(file+'*.bmp'):
#     im=ndimage.imread(filename,flatten=True)
#     feat = compute_features(im)
#     feat_list.append(feat)
#myarray = np.array(feat_list)
#
#np.savetxt("myarray.txt", myarray, delimiter=" ")
myarray_read= np.loadtxt('myarray.txt')
scores_read= np.loadtxt('FR_IQA_Scores.txt')


X=myarray_read
X_score= scores_read

file_Y="/Users/bingxinhou/Documents/SCU/* 338VideoComptression/TID2013/MOS.txt"
Y = np.loadtxt(file_Y)

X_select = X
Y_select = Y

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
rho_1, pval_1 = pearsonr(Y_pred,Y)
rho_2, pval_2 = spearmanr(Y_pred,Y)
rho_3, pval_3 = kendalltau(Y_pred,Y)
print('Method2: '+str([rho_1,rho_2,rho_3]))
#print('R2='+str(r2_score(Y_pred,Y)))
#print('MSE='+str(mean_squared_error(Y_pred,Y)))

#for n in range(9):
for n in range(1,14):
 rho1, pval1 = pearsonr(X_score[:,n],Y)
 rho2, pval2 = spearmanr(X_score[:,n],Y)
 rho3, pval3 = kendalltau(X_score[:,n],Y)
 print(dict[n+1]+str([rho1,rho2,rho3]))
# print(r2_score(X[:,n],Y))
# print(mean_squared_error(X[:,n],Y))







