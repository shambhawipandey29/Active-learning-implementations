# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:17:11 2020

@author: ss2528
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from modAL.models import ActiveLearner
from joblib import Parallel, delayed
from copy import deepcopy
import math
import time
from datetime import datetime

### dataset ###
#df=pd.read_csv("train.csv")
df=pd.read_csv("./dataset/LMTO_B_new.csv")
#df.drop(["name", "OH"], axis=1, inplace=True)
df.drop(["name", "OH", "Tolerance_factor", "Octahedral_factor", "ionic_ration_AO", "ionic_ration_BO", "average_ionic_radius1"], axis=1, inplace=True)

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
            estimator= ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0),
            #estimator= RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0),
            X_training=X_train_std, y_training=y_train
            )
  
  dummy_learner = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0)
  #dummy_learner = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0)
        
  
  n_queries = 273
  for idx in range(n_queries):
    y_rmse=[]
    if (idx==0):
        for j in range(len(y_train_err)):
          X_training = deepcopy(X_train_std)
          y_training = deepcopy(y_train)
          # removing jth instance
          X_training = np.delete(X_training, j, axis= 0)
          y_training = np.delete(y_training, j)
          # training on remaining instances
          dummy_learner.fit(X_training, y_training)
          # obtaining test error on jth instance
          y_pred = dummy_learner.predict([X_train_std[j]])
          y_rmse.append(abs(y_train[j]- y_pred[-1]))
        
        # residual model training on test error
        #rfr = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_rmse)
        rfr = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_rmse)
        
        y_expected = rfr.predict(X_pool)
        
        query_idx = np.argmax(y_expected)
        
        
    else:
        X_train_err = np.append(X_train_err, X_pool[query_idx].reshape(1, -1), axis=0)
    
        y_train_err = np.append(y_train_err, y_pool[query_idx])
        
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
        
        for j in range(len(y_train_err)):
          X_training = deepcopy(X_train_err)
          y_training = deepcopy(y_train_err)
          # removing jth instance
          X_training = np.delete(X_training, j, axis= 0)
          y_training = np.delete(y_training, j)
          # training on remaining instances
          dummy_learner.fit(X_training, y_training)
          #obtaining test error on remaining instances
          y_pred = dummy_learner.predict([X_train_err[j]])
          y_rmse.append(abs(y_train_err[j]- y_pred[-1]))
        
        #rfr = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_rmse)
        rfr = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0).fit(X_train_err, y_rmse)
        
        y_expected = rfr.predict(X_pool)  
        
        # teaching new instance
        main_learner.teach(
                        X = X_train_err[-1].reshape(1, -1),
                        y = y_train_err[-1].reshape(1, )
                        )
        
        
        rmse.append(math.sqrt(mean_squared_error(y_pool, main_learner.predict(X_pool))))
        mae.append((mean_absolute_error(y_pool, main_learner.predict(X_pool))))
        rmse_train.append(math.sqrt(mean_squared_error(y_train_err, main_learner.predict(X_train_err)))) 
        mae_train.append((mean_absolute_error(y_train_err, main_learner.predict(X_train_err))))
        mae_error_predictor.append(np.mean(rfr.predict(X_pool)))
        
        
        query_idx = np.argmax(y_expected)
    
        
    
        
    
  return rmse[-1], rmse_train[-1], mae[-1], mae_train[-1], mae_error_predictor[-1]


number_of_iterations = 100

results= Parallel(n_jobs=-1)(delayed(expected_error_active_learner)(X_train, Y_output, i) for i in range(number_of_iterations))
tresults = np.transpose(results)
print ("Mean test MAEs: {}".format(np.mean(tresults[2])))
print ("Standard deviation test MAEs: {}".format(np.std(tresults[2])))
print ("Mean test RMSEs: {}".format(np.mean(tresults[0])))
print ("Standard deviation test RMSE: {}".format(np.std(tresults[0])))
