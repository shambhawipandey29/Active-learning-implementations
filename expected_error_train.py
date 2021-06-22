# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:17:11 2020

@author: ss2528
"""

import numpy as np
import pandas as pd



#import matplotlib.pyplot as plt
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from joblib import Parallel, delayed
from copy import deepcopy
from modAL.models import ActiveLearner
import math
import time
from datetime import datetime

df=pd.read_csv("train.csv")
#df=pd.read_csv("LMTO_B.csv")
df.drop(["name", "OH"], axis=1, inplace=True)
#df.drop(["name", "OH", "Tolerance_factor", "Octahedral_factor", "ionic_ration_AO", "ionic_ration_BO", "average_ionic_radius1"], axis=1, inplace=True)


target_column = "O"
x_input = df.drop(target_column, axis = 1)
y_output = df[target_column]

std_scale = preprocessing.StandardScaler().fit(x_input)
X_train = std_scale.transform(x_input)
Y_output = pd.Series(y_output).values

def expected_error_active_learner(X_train, Y_output, i):
  print(datetime.now())
  time.sleep(5)
  print(datetime.now())
  rmse = []
  mae= []
  rmse_train = []
  mae_train = []
  mae_error_predictor = []
  X_pool = deepcopy(X_train)
  y_pool = deepcopy(Y_output)
  RANDOM_STATE_SEED = i
  np.random.seed(RANDOM_STATE_SEED)
  n_initial = 3
  train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
  X_train_std = X_pool[train_idx]
  y_train = y_pool[train_idx]

  X_train_err = X_pool[train_idx]
  y_train_err = y_pool[train_idx]

  # creating a reduced copy of the data with the known instances removed
  X_pool = np.delete(X_pool, train_idx, axis=0)
  y_pool = np.delete(y_pool, train_idx)
  print(len(X_pool))
  print((len(X_train_std)))
  
  # initializing learner
  main_learner = ActiveLearner(
            #estimator= ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0),
            estimator= RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0),
            #estimator=GaussianProcessRegressor(kernel=kernel),
            X_training=X_train_std, y_training=y_train
            )       
  
  n_queries = 273

  
  for idx in range(n_queries):
    
        y_train_rmse=[]
        
        for j in range(len(y_train_err)):
          #obtaining train error on jth instances
          y_train_pred = main_learner.predict([X_train_err[j]])
          y_train_rmse.append(abs(y_train_err[j]- y_train_pred[-1]))
        # training the residual model on training error of labelled instances
        rfr = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_train_rmse)
        #rfr = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_train_rmse)
        
        y_expected = rfr.predict(X_pool)
        
        query_idx = np.argmax(y_expected)
        
        # add queried instance to train
        X_train_err = np.append(X_train_err, X_pool[query_idx].reshape(1, -1), axis=0)
    
        y_train_err = np.append(y_train_err, y_pool[query_idx])
        
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        main_learner.teach(
                        #X = X_train_err,
                        #y = y_train_err
                        X = X_train_err[-1].reshape(1, -1),
                        y = y_train_err[-1].reshape(1, )
                        #X=X_pool[query_idx].reshape(1, -1),
                        #y=y_pool[query_idx].reshape(1, )
                        )
        
        
        rmse.append(math.sqrt(mean_squared_error(y_pool, main_learner.predict(X_pool))))
        mae.append((mean_absolute_error(y_pool, main_learner.predict(X_pool))))
        rmse_train.append(math.sqrt(mean_squared_error(y_train_err, main_learner.predict(X_train_err)))) 
        mae_train.append((mean_absolute_error(y_train_err, main_learner.predict(X_train_err))))
        mae_error_predictor.append(np.mean(rfr.predict(X_pool)))
        
    
        
  return rmse[-1], rmse_train[-1], mae[-1], mae_train[-1], mae_error_predictor[-1]



number_of_iterations = 100

results= Parallel(n_jobs=-1)(delayed(expected_error_active_learner)(X_train, Y_output, i) for i in range(number_of_iterations))
tresults = np.transpose(results)
print ("Mean test RMSEs: {}".format(np.mean(tresults[0])))
print ("Standard deviation test RMSEs: {}".format(np.std(tresults[0])))
print ("Mean training RMSEs: {}".format(np.mean(tresults[1])))
print ("Standard deviation training RMSEs: {}".format(np.std(tresults[1])))
print("Max training RMSE: {}".format(max(tresults[1])))
print ("Mean testing RMSEs: {}".format(np.mean(tresults[0])))
print ("Standard deviation testing RMSEs: {}".format(np.std(tresults[0])))
print("Max testing RMSE: {}".format(max(tresults[0])))
print("Min testing RMSE: {}".format(min(tresults[0])))
print ("Mean training MAEs: {}".format(np.mean(tresults[3])))
print ("Standard deviation training MAEs: {}".format(np.std(tresults[3])))
print("Max training MAE: {}".format(max(tresults[3])))
print ("Mean testing MAEs: {}".format(np.mean(tresults[2])))
print ("Standard deviation testing MAEs: {}".format(np.std(tresults[2])))
print("Max testing MAE: {}".format(max(tresults[2])))
print("Min testing MAE: {}".format(min(tresults[2])))