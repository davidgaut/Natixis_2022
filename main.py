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

from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error

def loss(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def loss_for_lgbm(preds: np.ndarray, data: Dataset, threshold: float=0.5,): #-> Tuple[str, float, bool]:
    """Calculate Binary Accuracy"""
    y_true = data.get_label()
    acc = np.sqrt(mean_squared_error(y_true, preds))
    return 'custom_loss', acc, False

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
sns.lineplot(data=targets,x=targets.index,y=targets.Diff_EURUSD_2w)
#%% Linear Model
X = embeddings.values
y = targets.values

X_train,X_test,y_train,y_test = train_test_split(X,y)

# %% Basic Test
means = np.mean(y_train,axis=0)
y_pred_mean = 0*y_test + means
print('Error is',loss(y_pred_mean,y_test))
# %% Simple Multi Output regression
regr  = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train,y_train)
ypred = regr.predict(X_test)
print(np.shape(ypred))
print('Error is',loss(ypred,y_test))

# %% Gradient Boosting
best_params_ = dict()
for i in range(n_features):
    y_train_      = y_train.T[i].reshape((-1,1))
    train_dataset = Dataset(X_train, label=y_train_, free_raw_data=False, )

    options['default_param'] = "{'boosting' :'gbdt', 'objective': 'mse', 'learning_rate':0.02, 'early_stopping_rounds':150, 'zero_as_missing' : True, 'force_col_wise':True, 'verbose': -1,}"

    X_idx      = range(0,X_train.shape[0])
    objective_ = lambda trial: objective_cv(trial,train_dataset,TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx),eval(options['default_param']),metrics=loss_for_lgbm)
    if new:
        study = optuna.create_study(direction="minimize")
    else:
        study = joblib.load('./predictors/lgbm_{:}.pkl'.format(i))

    study.optimize(objective_, n_trials=20, timeout=600, n_jobs=-1)

    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ",   len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params_.update({i:trial.params})
    joblib.dump(study,'./predictors/lgbm_{:}.pkl'.format(i))

# %% Prediction
embeddings_test = pd.read_csv('x_test_pf4T2aK.csv').iloc[:,:max_feat]

X_test = embeddings_test.values
if False:
    y_test = regr.predict(X_test)
    print(np.shape(y_test))
    prediction = pd.DataFrame(y_test,columns=variables)

# Best param set
# Refit
targets_pred = pd.DataFrame(columns=variables)
for i in range(n_features):
    optim_param = best_params_[i]
    optim_param.update(eval(options['default_param']))
    optim_param['num_boost_round'] = 10000
    y_train_ = y_train.T[i].reshape((-1,1))
        
    TS_fold = TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx)
    cv_results  = cv(optim_param, train_dataset, folds=TS_fold, 
            num_boost_round=10000, \
            shuffle=False, #fpreproc=regfit,
            stratified=False, seed=2021, 
            feval=loss_for_lgbm,
            early_stopping_rounds=1, verbose_eval=200, 
            eval_train_metric=False, return_cvbooster=False)

    optim_param['num_boost_round']       = round(1.1*np.max((np.argmin(cv_results['custom_loss-mean']),1))) 
    optim_param['early_stopping_rounds'] = None

    final_model  = train(optim_param, train_dataset,)
    targets_pred.loc[:,variables[i]] = final_model.predict(X_test)
    
prediction = targets_pred

# %% [markdown] 
# Optimization Plots
# %%
plot_optimization_history(study)
# %%
plot_parallel_coordinate(study)


# %%
id_ = datetime.strftime(datetime.now(),'%Y-%b-%d-%m-%s')
if True:
    print('Saving file.')
    prediction.to_csv('./predictions/prediction_linear_'+id_)
# %%
