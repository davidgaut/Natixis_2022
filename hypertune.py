import neptune, optuna, stats
import neptunecontrib.monitoring.optuna as opt_utils
from lightgbm import reset_parameter, cv, train
from numpy import argmin, reshape
from pandas import DataFrame

def init(options):
    '''Init connexion to Neptune...'''
    try: 
        projects = neptune.init(project_qualified_name='davidgaut/'+options['name'], api_token='put_your_neptune_apikey')
    except ValueError:
        print('NoConnexion to Neptune')
        projects = None
    return projects


def log_new(projects,NewEstim,options):
    '''Start new study with Optuna / Neptune,
     - params init to log'''

    study = optuna.create_study(direction='minimize')
    if NewEstim:
        try:
            neptune.create_experiment(
                'lgb-tuning', 
                upload_source_files=options["source_files"], 
                tags=[options["tags"]],
                params=eval(options['default_param']))
            neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)
        except ValueError:
            print('NoConnexion to Neptune')
            print(ValueError)
            neptune_callback = None
    else:
        import joblib
        # my_exp = [exp for exp in projects.get_experiments() if str(exp).endswith('239'+')')][0]
        my_exp = [exp for exp in projects.get_experiments()][-1]
        print('Using experiment -> ',str(my_exp))
        my_exp.download_artifact('study.pkl')
        study = joblib.load("study.pkl")
        print("Number of Trials: "+str(len(study.trials)))
        print("Best trial until now:\nValue: ", study.best_trial.value)
        print(" Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        neptune_callback = opt_utils.NeptuneCallback(experiment=my_exp,log_study=True, log_charts=True)
    return study, neptune_callback


def objective_cv(trial,train_dataset,TS_fold,params_fix,metrics=None,callbacks=None):
    param = {
        'objective':        trial.suggest_categorical('objective',['mse','huber','mae']),
        'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-10, 5.0),
        'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-10, 5.0),
        'num_leaves':       trial.suggest_int('num_leaves', 10, 1000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.60),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 25),

        'max_cat_threshold':trial.suggest_int('max_cat_threshold', 5,800),
        'scale_pos_weight' :trial.suggest_int('scale_pos_weight', 1,1e4),
        'cat_smooth'       :trial.suggest_uniform('cat_smooth', 1.0,100.0),
        'max_bin':          trial.suggest_int('max_bin', 100, 1800),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 500),
        'min_child_weight' :trial.suggest_uniform('min_child_weight',  1e-5,1e3),
        'min_gain_to_split':trial.suggest_loguniform('min_gain_to_split', 1e-10,200.0),
    }
    
    params_fix.update(param)
    model = cv(params_fix, train_dataset, 
        folds=TS_fold, shuffle=False, stratified=False, seed=2021, 
        verbose_eval=200, num_boost_round=10000, 
        feval=metrics,
        callbacks=callbacks, 
        eval_train_metric=False, return_cvbooster=False,)  
    
    return model['custom_loss-mean'][-1] #+ model['l2-stdv'][-1]

def objective_valid(trial,train_dataset,valid_set,params_fix,metrics=None,callbacks=None):
    param = {
        'objective':        trial.suggest_categorical('objective',['mse','huber','mae']),
        'lambda_l1':        trial.suggest_loguniform('lambda_l1', 1e-10, 5.0),
        'lambda_l2':        trial.suggest_loguniform('lambda_l2', 1e-10, 5.0),
        'num_leaves':       trial.suggest_int('num_leaves', 10, 1000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.60),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 25),

        'max_cat_threshold':trial.suggest_int('max_cat_threshold', 5,800),
        'scale_pos_weight' :trial.suggest_int('scale_pos_weight', 1,1e4),
        'cat_smooth'       :trial.suggest_uniform('cat_smooth', 1.0,100.0),
        'max_bin':          trial.suggest_int('max_bin', 100, 1800),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 500),
        'min_child_weight' :trial.suggest_uniform('min_child_weight',  1e-5,1e3),
        'min_gain_to_split':trial.suggest_loguniform('min_gain_to_split', 1e-10,200.0),
    }
    
    params_fix.update(param)
    model  = train(params_fix, train_dataset, valid_sets=[valid_set],
        verbose_eval=200, num_boost_round=10000, 
        feval=metrics,
        callbacks=callbacks, 
    )  

    return model.best_score['valid_0']['metric']

def predict(study,train_dataset,TS_fold,resid,X_test,options):
    # Best param set
    optim_param = study.best_params
    optim_param.update(eval(options['default_param']))

    # Refit
    cv_results  = cv(optim_param, train_dataset, folds=TS_fold, 
            num_boost_round=10000, \
            shuffle=False, #fpreproc=regfit,
            stratified=False, seed=2021, 
            early_stopping_rounds=1, verbose_eval=100, 
            eval_train_metric=False, return_cvbooster=False)

    optim_param['num_boost_round']       = round(1.1*argmin(cv_results['l2-mean'])) 
    optim_param['early_stopping_rounds'] = None

    final_model = train(optim_param, train_dataset,)
        
    prediction = final_model.predict(X_test) + reshape(resid,(len(resid),))
    submission = DataFrame({'target' : prediction},index=X_test.index); print('Saving Results..')
    submission.to_csv(options['folder_path']+'/results.csv')