# %%
import sys, os
sys.path.append('/home/davidg/Codes/Python_Codes/Projects/Hackatons/CFM_data_challenge')
import joblib
import json
from hypertune import objective_cv
from typing import Tuple
import optuna

from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history
from optuna.trial import TrialState
from lightgbm import cv, train, Dataset

import pandas as pd
import seaborn as sns
import numpy as np

from joblib import Parallel, delayed
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib as mpl
mpl.style.use('seaborn')

def loss(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def loss_for_lgbm(preds: np.ndarray, data: Dataset):
    """Calculate Binary Accuracy"""
    y_true = data.get_label()
    loss   = np.sqrt(mean_squared_error(y_true, preds))
    return 'custom_loss', loss, False

with open('options.json') as f:
    options = json.load(f)
    print(json.dumps(options, indent=4, sort_keys=True))

#%% Load data
test = True
new  = True 

os.makedirs('./predictors/',exist_ok=True)

if test:
    max_feat = 50
else:
    max_feat = 768

embeddings = pd.read_csv('x_train_ACFqOMF.csv').iloc[:,:max_feat]
targets    = pd.read_csv('y_train_HNMbC27.csv',index_col=0)
variables  = targets.columns.to_list()
n_features = np.shape(targets)[1]

#%% Plot
sns.lineplot(data=targets,)#x=targets.index,y=targets.columns)
#%% Targets and Splits
X = embeddings.values
y = targets.values

X_train,X_test,y_train,y_test = train_test_split(X,y)

# %% Basic Test
means = np.mean(y_train,axis=0)
y_pred_mean = 0*y_test + means
print('Error targeting mean is',loss(y_pred_mean,y_test))
# %% Simple Multi Output regression
regr  = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train,y_train)
ypred = regr.predict(X_test)
print(np.shape(ypred))
print('Error with multiouptut linear regression is',loss(ypred,y_test))

# %% Gradient Boosting
def lgb_multioutput_tune(id_,X_train:pd.DataFrame,y_train:np.array,options:dict,verbose=False):
    '''
    To tune lgb with multitarget.
     - id_ (str): target id
     - X_train (pd.DataFrame): features
     - y_train (np.array): target
     - options (dict): estimation options
     - verbose (bool): to display results
    '''
    train_dataset = Dataset(X_train, label=y_train, free_raw_data=False, )
    X_idx         = range(0,X_train.shape[0])
    objective_    = lambda trial: objective_cv(trial,train_dataset,TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx),eval(options['default_param']),metrics=loss_for_lgbm)

    if new:
        study = optuna.create_study(direction="minimize")
    else:
        study = joblib.load('./predictors/lgbm_{:}.pkl'.format(id_))

    study.optimize(objective_, n_trials=20, timeout=10,)
    trial = study.best_trial
    if verbose:
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    joblib.dump(study,'./predictors/lgbm_{:}.pkl'.format(i))
    return {id_:trial.params}


def lgb_multioutput_predict(id_,best_params_,X_test,X_train:pd.DataFrame,y_train,options:dict):
    '''
    To tune lgb with multitarget.
     - id_ (str): target id
     - X_train (pd.DataFrame): features
     - y_train (np.array): target
     - options (dict): estimation options
     - verbose (bool): to display results
    '''
    optim_param = best_params_[id_]
    optim_param.update(eval(options['default_param']))
    optim_param['num_boost_round'] = 10000
    
    train_dataset = Dataset(X_train, label=y_train, free_raw_data=False, )
    
    X_idx   = range(0,X_train.shape[0])
    TS_fold = TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx)
    cv_results  = cv(optim_param, train_dataset, folds=TS_fold, 
            num_boost_round=10000, \
            shuffle=False, #fpreproc=regfit,
            stratified=False, seed=2021, 
            feval=loss_for_lgbm,
            early_stopping_rounds=1, verbose_eval=200, 
            eval_train_metric=False, return_cvbooster=False)

    optim_param['num_boost_round']       = round(1.0*np.max((np.argmin(cv_results['custom_loss-mean']),1))) 
    optim_param['early_stopping_rounds'] = None

    final_model = train(optim_param, train_dataset,)
    y_pred      = final_model.predict(X_test)
    return {id_:y_pred}

lgb_tune_loop = lambda ii: lgb_multioutput_tune(variables[ii],X_train,y_train.T[ii].reshape((-1,1)),options)
results       = Parallel(n_jobs=2)(delayed(lgb_tune_loop)(ii) for ii in range(n_features))

best_params_ = dict()
for d in results:
    best_params_.update(d) 

# %% Prediction
embeddings_test = pd.read_csv('x_test_pf4T2aK.csv').iloc[:,:max_feat]

X_test = embeddings_test.values
if False:
    y_test = regr.predict(X_test)
    print(np.shape(y_test))
    prediction = pd.DataFrame(y_test,columns=variables)


lgb_predict_loop = lambda ii: lgb_multioutput_predict(variables[ii],best_params_,X_test,X_train,y_train.T[ii].reshape((-1,1)),options)
predictions      = Parallel(n_jobs=2)(delayed(lgb_predict_loop)(ii) for ii in range(n_features))

all_pred = dict()
for d in predictions:
    all_pred.update(d) 

y_pred = pd.DataFrame(all_pred)

# %% [markdown] 
# # Optimization Plots
# # %%
# plot_optimization_history(study)
# # %%
# plot_parallel_coordinate(study)


# %%
id_ = datetime.strftime(datetime.now(),'%Y-%b-%d-%m-%s')
if True:
    print('Saving file.')
    prediction.to_csv('./predictions/prediction_linear_'+id_)
# %%
