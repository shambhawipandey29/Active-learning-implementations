# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:23:23 2020

@author: ss2528
"""

import numpy as np
import pandas as pd
#from scipy import stats

import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFECV
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
from joblib import Parallel, delayed
import math
from sklearn.feature_selection import RFECV
import time
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA

### dataset ###
#df=pd.read_csv("train.csv")
df=pd.read_csv("LMTO_B_new.csv")
#df.drop(["name", "OH"], axis=1, inplace=True)
df.drop(["name", "OH", "Tolerance_factor", "Octahedral_factor", "ionic_ration_AO", "ionic_ration_BO", "average_ionic_radius1"], axis=1, inplace=True)

target_column = "O"
x_input = df.drop(target_column, axis = 1)
y_output = df[target_column]

std_scale = preprocessing.StandardScaler().fit(x_input)
X_train = std_scale.transform(x_input)
Y_output = pd.Series(y_output).values


def committee_active_learner_outliers(X_train, Y_output, x_input, i):
  print(datetime.now())
  time.sleep(5)
  print(datetime.now())
  rmse = []
  mae = []
  rmse_train =[]
  mae_train = []
  rmse_pool = []
  mae_pool = []
  
  std_scale = preprocessing.StandardScaler().fit(X_train)
  X_train_std = std_scale.transform(X_train)
  
  
  
  Q1 = x_input.quantile(0.25)
  Q3 = x_input.quantile(0.75)
  IQR = Q3 - Q1
  bl = ((x_input < (Q1 - 1.5 * IQR)) |(x_input > (Q3 + 1.5 * IQR)))
  ans = bl.apply(np.sum , axis =1)
  mylist = []
  for index, value in ans.items():
      if( value > 3 ):
          mylist.append(index)

  X_outlier = [deepcopy(X_train_std[mylist[0]])]
  y_outlier = [deepcopy(Y_output[mylist[0]])]
  
  X_label = deepcopy(X_train_std)
  y_label = deepcopy(Y_output)
  
  for j in range(1, len(mylist)):
    
        X_outlier = np.append(X_outlier, X_train_std[mylist[j]].reshape(1, -1), axis=0)
        y_outlier = np.append(y_outlier, Y_output[mylist[j]])
  X_label = np.delete(X_train, mylist, axis=0)
  y_label = np.delete(Y_output, mylist) 
  
  
  RANDOM_STATE_SEED = i
  np.random.seed(RANDOM_STATE_SEED)
  
  #### initializing the regressors ###
  n_initial = 3
  
  initial_idx = list()
  initial_idx.append(np.random.choice(range(len(X_outlier)), size=n_initial, replace=False))
  #initial_idx.append(np.random.choice(range(122, 244), size=n_initial, replace=False))
  
  learner_list=[]
  X_training=[]
  y_training=[]
  
  
  X_training = X_outlier[initial_idx[0]]
  y_training = y_outlier[initial_idx[0]]
  
  #extra_regressor = ExtraTreesRegressor(n_estimators= 50, min_samples_split=4, random_state=0)
  #extra_regressor = RandomForestRegressor(n_estimators= 50, min_samples_split=4, random_state=0)
  
  learner_list.append(ActiveLearner(
                      estimator = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0),
                      #estimator = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0),
                      #estimator = RFECV(extra_regressor, cv=4, scoring= 'neg_mean_squared_error'),
                      X_training=X_outlier[initial_idx[0]], y_training=y_outlier[initial_idx[0]]
                      ))
  
  X_outlier = np.delete(X_outlier, initial_idx[0], axis=0)
  y_outlier = np.delete(y_outlier, initial_idx[0]) 
  
  initial_idx.append(np.random.choice(range(len(X_outlier)), size=n_initial, replace=False))
  #initial_idx.append(np.random.choice(range(len(X_label)), size=n_initial, replace=False))

  X_training = np.append(X_training, X_label[initial_idx[1]], axis=0)
  y_training = np.append(y_training, y_label[initial_idx[1]])
  
  
  learner_list.append(ActiveLearner(
                        #estimator=GaussianProcessRegressor(kernel, alpha=0.0, n_restarts_optimizer=20),
                        estimator = ExtraTreesRegressor(n_estimators=50, min_samples_split=3, random_state=0),
                        #estimator = RandomForestRegressor(n_estimators=50, min_samples_split=3, random_state=0),
                        #estimator = RFECV(extra_regressor, cv=4, scoring= 'neg_mean_squared_error'),
                        X_training=X_outlier[initial_idx[1]], y_training=y_outlier[initial_idx[1]]
                        #X_training=X_label[initial_idx[1]], y_training=y_label[initial_idx[1]]
                        ))
  
  #X_label = np.delete(X_label, initial_idx[1], axis=0)
  #y_label = np.delete(y_label, initial_idx[1])
  X_outlier = np.delete(X_outlier, initial_idx[1], axis=0)
  y_outlier = np.delete(y_outlier, initial_idx[1])
  
    
  X_pool = np.concatenate((X_outlier, X_label), axis=0)
  y_pool = np.concatenate((y_outlier, y_label), axis=0)
  ### initializing the Committee ###
  committee = CommitteeRegressor(
              learner_list=learner_list,
              query_strategy=max_std_sampling
              )

  ### active learning by iteration ###
  n_queries = 30
  batch_size = 9
  
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
          ### add queried instances to training ###
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
      
      
      committee.teach(X_batch, y_batch)
      iteration.append(idx+1)
     

      
  return rmse_pool, rmse, mae_pool, mae, rmse_train,  mae_train, initial_idx, iteration


number_of_iterations = 100
results = Parallel(n_jobs=-1)(delayed(committee_active_learner_outliers)(X_train, Y_output, x_input, i) for i in range(number_of_iterations))

rmse = []
rmse_pool = []
rmse_forecast =[]

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
diff = []
diff_mae = []

for i in range(number_of_iterations):
    rmse_final.append(rmse[i][-1])
    rmse_final_pool.append(rmse_pool[i][-1])
    mae_final.append(mae[i][-1])
    mae_final_pool.append(mae_pool[i][-1])
    rmse_train_final.append(rmse_train[i][-1])
    mae_train_final.append(mae_train[i][-1])
    
### forecasting model ###
n_queries = 30
prediction_idx = 19
model_2 = ARIMA(rmse[np.argmax(rmse_final_pool)], order=(3, 1, 3))
model_fit_2 = model_2.fit(method='css')
rmse_forecast = model_fit_2.predict(start=prediction_idx+1, end=n_queries, typ='levels')
for e in range(prediction_idx):
  rmse_forecast = np.insert(rmse_forecast, 0, None, axis = 0)

###ploting RMSE/MAE #######
colour=['b', 'g', 'r', 'purple']
fig_rmse_max= plt.figure()
plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse[np.argmax(rmse_final_pool)], '-.', c=colour[0],)
plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse_pool[np.argmax(rmse_final_pool)], c=colour[1])
plt.plot(iteration[np.argmax(rmse_final_pool)],  rmse_forecast, c=colour[2])
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
