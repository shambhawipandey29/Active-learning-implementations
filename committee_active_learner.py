# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:04:21 2020

@author: shambhawi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
from joblib import Parallel, delayed
from copy import deepcopy
import math
from sklearn.feature_selection import RFECV
import time
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

### dataset ###
#df=pd.read_csv("./dataset/train.csv")
df=pd.read_csv("./dataset/LMTO_B_new.csv")
#df.drop(["name", "OH"], axis=1, inplace=True)
df.drop(["name", "OH", "U_param", "spin","Tolerance_factor", "Octahedral_factor", "ionic_ration_AO", "ionic_ration_BO", "average_ionic_radius1"], axis=1, inplace=True)

target_column = "O"
x_input = df.drop(target_column, axis = 1)
y_output = df[target_column]

std_scale = preprocessing.StandardScaler().fit(x_input)
X_train = std_scale.transform(x_input)
Y_output = pd.Series(y_output).values

def committee_active_learner(X_train, Y_output, i):
  print(datetime.now())
  time.sleep(5)
  print(datetime.now())
  rmse = []
  mae = []
  rmse_train =[]
  mae_train = []
  rmse_pool = []
  mae_pool = []
  #rankings_1=[]
  #rankings_2=[]
  
  X_pool = deepcopy(X_train)
  y_pool = deepcopy(Y_output)
  
  X_train_std = deepcopy(X_train)
  y_train_std = deepcopy(Y_output)
  
  RANDOM_STATE_SEED = i
  np.random.seed(RANDOM_STATE_SEED)
  
  # initializing the regressors
  n_initial = 3
  

  initial_idx = list()
  initial_idx.append(np.random.choice(range(184), size=n_initial, replace=False))
  #initial_idx.append(np.random.choice(range(122, 244), size=n_initial, replace=False))
  initial_idx.append(np.random.choice(range(184, X_train.shape[0]), size=n_initial, replace=False))

  learner_list=[]
  X_training=[]
  y_training=[]
  count=0
  for idx in initial_idx:
    
    if count==0:
      X_training = X_train_std[idx]
      y_training = y_train_std[idx]
    
    else:
      X_training = np.append(X_training, X_train_std[idx], axis=0)
      y_training = np.append(y_training, y_train_std[idx])
    
    print(X_training.shape)
    print(y_training.shape)
    #regressor = ExtraTreesRegressor(n_estimators= 50, min_samples_split=3, random_state=0)
    #regressor = RandomForestRegressor(n_estimators= 50, min_samples_split=4, random_state=0)
    
    learner_list.append(ActiveLearner(
                        estimator = ExtraTreesRegressor(n_estimators= 50, min_samples_split=3, random_state=0),
                        #estimator = RFECV(regressor, cv=4, scoring= 'neg_mean_squared_error'), ### uncomment below for recursive feature elimination ####
                        #estimator = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0),
                        X_training=X_train_std[idx], y_training=y_train_std[idx]
                        ))
    
    
    
    print(Y_output[idx])
    print(y_training)
    count = count+1
    
  X_pool = np.delete(X_pool, initial_idx, axis=0)
  y_pool = np.delete(y_pool, initial_idx)
  
  ### initializing the Committee ###
  committee = CommitteeRegressor(
              learner_list=learner_list,
              query_strategy=max_std_sampling
              )

  #### active regression ####
  n_queries = 270
  batch_size = 1
  iteration = []
  for idx in range(n_queries):
      for h in range(batch_size):
          query_idx, query_instance = committee.query(X_pool)
          
          if h==0:
              X_batch = X_pool[query_idx]
              y_batch = y_pool[query_idx]
              print(X_batch.shape)
              print(y_batch.shape)
          else:
              X_batch = np.append(X_batch, X_pool[query_idx].reshape(1, -1), axis=0)
              y_batch = np.append(y_batch, y_pool[query_idx])
          # add queried instances to training
          X_training = np.append(X_training, X_pool[query_idx].reshape(1, -1), axis=0)
          y_training = np.append(y_training, y_pool[query_idx])
          #print(X_training.shape)
          #print(y_training.shape)
          
          ### remove queried instance from pool ###
          X_pool = np.delete(X_pool, query_idx, axis=0)
          y_pool = np.delete(y_pool, query_idx)
      print(X_batch.shape)
      print(y_batch.shape)
      
      rmse.append(math.sqrt(mean_squared_error(y_batch, committee.predict(X_batch)))) 
      mae.append((mean_absolute_error(y_batch, committee.predict(X_batch))))
      rmse_train.append(math.sqrt(mean_squared_error(y_training, committee.predict(X_training)))) 
      mae_train.append((mean_absolute_error(y_training, committee.predict(X_training))))
      rmse_pool.append(math.sqrt(mean_squared_error(y_pool, committee.predict(X_pool))))
      mae_pool.append(mean_absolute_error(y_pool, committee.predict(X_pool))  )
      #rankings_1.append(committee.learner_list[0].estimator.ranking_)
      #rankings_2.append(committee.learner_list[1].estimator.ranking_)
      
      
      committee.teach(X_batch, y_batch)
      iteration.append(idx+1)
  
  return rmse_pool, rmse, mae_pool, mae, rmse_train,  mae_train, initial_idx, iteration

def forecasting(n_queries, prediction_idx):
    model_2 = ARIMA(rmse[np.argmax(rmse_final_pool)], order=(3, 1, 3))
    model_fit_2 = model_2.fit(method='css')
    rmse_forecast = model_fit_2.predict(start=prediction_idx+1, end=n_queries, typ='levels')
    for e in range(prediction_idx):
        rmse_forecast = np.insert(rmse_forecast, 0, None, axis = 0)    
    return rmse_forecast

number_of_iterations = 100
results = Parallel(n_jobs=-1)(delayed(committee_active_learner)(X_train, Y_output, i) for i in range(number_of_iterations))

rmse = []
rmse_pool = []
rmse_forecast =[]
mae_forecast=[]
mae = []
mae_pool = []
rmse_train =[]
mae_train = []
initial_idx = []
iteration = []

for i in range(number_of_iterations):
    rmse_pool.append(results[i][0])
    rmse.append(results[i][1])
    mae_pool.append(results[i][2])
    mae.append(results[i][3])
    rmse_train.append(results[i][4])
    mae_train.append(results[i][5])
    initial_idx.append(results[i][6])
    iteration.append(results[i][7])
    
rmse_final = []
rmse_final_pool = []
mae_final = []
mae_final_pool = []
rmse_train_final =[]
mae_train_final = []


for i in range(number_of_iterations):
    rmse_final.append(rmse[i][-1])
    rmse_final_pool.append(rmse_pool[i][-1])
    mae_final.append(mae[i][-1])
    mae_final_pool.append(mae_pool[i][-1])
    rmse_train_final.append(rmse_train[i][-1])
    mae_train_final.append(mae_train[i][-1])
    
###uncomment for forecasting ######
#rmse_forecast = forecasting(30, 19)

###ploting RMSE/MAE #######
colour=['b', 'g', 'r', 'purple']
fig_rmse_max= plt.figure()
plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse[np.argmax(rmse_final_pool)], '-.', c=colour[0],)
plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse_pool[np.argmax(rmse_final_pool)], c=colour[1])
#plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse_forecast, c=colour[2]) ##uncomment for forecasting
plt.xlabel("Number of iterations", fontsize= 20, fontname='Times New Roman')
plt.ylabel("RMSE (eV)", fontsize= 20, fontname='Times New Roman')
plt.title("Batch Size = 9", fontsize= 24, fontname='Times New Roman')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


fig_rmse_min= plt.figure()
plt.plot(iteration[np.argmin(rmse_final_pool)], rmse[np.argmin(rmse_final_pool)], '-.', c=colour[0])
plt.plot(iteration[np.argmin(rmse_final_pool)], rmse_pool[np.argmin(rmse_final_pool)], c=colour[1])
plt.xlabel("Number of iterations", fontsize= 20, fontname='Times New Roman')
plt.ylabel("RMSE (eV)", fontsize= 20, fontname='Times New Roman')
plt.title("Batch Size = 9", fontsize= 24, fontname='Times New Roman')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


fig_mae_max= plt.figure()
plt.plot(mae[np.argmax(mae_final_pool)],  '-.', c=colour[0],)
plt.plot(mae_pool[np.argmax(mae_final_pool)], c=colour[1])
plt.xlabel("Number of iterations", fontsize= 20, fontname='Times New Roman')
plt.ylabel("MAE (eV)", fontsize= 20, fontname='Times New Roman')
plt.title("Batch Size = 9", fontsize= 24, fontname='Times New Roman')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
    

fig_mae_min= plt.figure()
plt.plot(mae[np.argmin(mae_final_pool)], '-.', c=colour[0])
plt.plot(mae_pool[np.argmin(mae_final_pool)], c=colour[1])
plt.xlabel("Number of iterations",  fontsize= 20, fontname='Times New Roman')
plt.ylabel("MAE (eV)",  fontsize= 20, fontname='Times New Roman')
plt.title("Batch Size = 9", fontsize= 24, fontname='Times New Roman')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

print ("Mean testing MAEs: {}".format(np.mean(mae_final_pool)))
print ("Standard deviation testing MAEs: {}".format(np.std(mae_final_pool)))
print ("Mean testing RMSEs: {}".format(np.mean(rmse_final_pool)))
print ("Standard deviation testing RMSEs: {}".format(np.std(rmse_final_pool)))