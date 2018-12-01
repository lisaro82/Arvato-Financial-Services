from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold  as CVFold

from hyperopt import hp, tpe, fmin, space_eval, Trials

import datetime
import os

class CustBoostClassifier():
    
    __modelType__             = None
    __modelName__             = None,
    __models__                = None
    
    __X_train__               = None
    __y_train__               = None
    __feature_name__          = None
    
    __selected_featuresIni__  = None
    __imputeColumnsIni__      = None
    __scaleColumnsIni__       = None
    
    __random_state__          = None
    
    __cvSplits__              = None
    
    __dataset_train__         = None
    __dataset_valid__         = None
    __validation__            = None
    __cvScores__              = None
    __validScores__           = None
    
    
    #----------------------------------------------------------------------------------------------------------------
    def __init__( self, p_modelType, p_modelName = None, X_train = None, y_train = None,
                  p_selected_features = [], p_imputeColumns = [], p_scaleColumns = [],
                  p_cvSplits = 0, p_random_state = 0 ):
        self.__modelType__            = p_modelType
        self.__modelName__            = p_modelName
        self.__X_train__              = X_train
        self.__y_train__              = y_train
        self.__cvSplits__             = p_cvSplits
        
        self.__selected_featuresIni__ = p_selected_features   
        self.__imputeColumnsIni__     = p_imputeColumns 
        self.__scaleColumnsIni__      = p_scaleColumns
        
        self.__random_state__         = p_random_state
        self.__dataset_train__        = []
        
        if not X_train is None:
            self.__createCVSplits__()
        
        print(f'Class has been initialized for model type: {self.__modelType__}')        
        return
    
    
    #----------------------------------------------------------------------------------------------------------------
    def train(self, p_params, p_verbose = False, p_saveFull = True):
        """ Generic function to train models. It will call the iternal functions basd on the model type. """ 
        if self.__modelType__ == 'lightgbm':
            self.__trainLGB__(p_params, p_saveFull, p_verbose)
        else:
            raise ValueError(f'Training not implemented for model type: {self.__modelType__}')
        return
    
    
    #----------------------------------------------------------------------------------------------------------------
    def predictStack(self, p_folder, X_data, y_data = None, p_showPlot = False):
        """ Function used to load all the models that will be used to create the predictions. """ 
        for folder in p_folder:
            for (dirpath, dirnames, filenames) in os.walk(f'models/stack/{folder}'):
                for filename in filenames:    
                    bst = lgb.Booster(model_file = filename)
                    self.__models__.append(bst)
        return self.predictProba(X_data, y_data, p_showPlot)
    
    
    #----------------------------------------------------------------------------------------------------------------
    def predictProba(self, X_data, y_data = None, p_showPlot = False):  
        """ Returns the mean predictions for the models linked to the current instance of the class. """  
        X_data = self.__scaleData__(X_data)
        
        y_pred = None
        for model in self.__models__:
            if self.__modelType__ == 'lightgbm':
                y_predTmp = model.predict(X_data)  
            else:
                y_predTmp = model.predict_proba(X_data)[:, 1] 
                
            if y_pred is None: y_pred = y_predTmp.reshape(-1, 1)
            else: y_pred = np.append(y_pred, y_predTmp.reshape(-1, 1), axis = 1)
        
        y_pred = np.mean(y_pred, axis = 1)
        
        if not y_data is None:
            print(f'   ROC / AUC score: {round(roc_auc_score(y_data, y_pred), 6)}.')            
            if p_showPlot:
                v_data = pd.DataFrame({ "True":       y_data.values,
                                        "Predicted":  np.round(y_pred, 2) }, index = y_data.index)
                ax = sns.countplot(y = "Predicted", data = v_data[v_data["True"] == 0])
                plt.show()
                ax = sns.countplot(y = "Predicted", data = v_data[v_data["True"] == 1])
                plt.show()                
                display(v_data[v_data["True"] != 0].head())
                
        return y_pred
    
    
    #----------------------------------------------------------------------------------------------------------------
    def getScore(self):
        """ Returns the validation mean validation score for the models linked to the current instance of the class. """ 
        return np.mean(self.__validScores__)
    
    
    #----------------------------------------------------------------------------------------------------------------
    def __trainLGB__(self, p_params, p_sufix = 'FULL', p_saveFull = True, p_verbose = False):
        """ Train LGB models. """ 
        self.__models__      = []
        self.__validScores__ = []
        
        v_params = p_params.copy()
        v_params['random_state'] = self.__random_state__
        if 'n_estimators' in p_params.keys():
            v_num_boost_round = p_params['n_estimators']
            v_params.pop('n_estimators', None)
        else:
            v_num_boost_round = 3000
                            
        for idx in range(len(self.__dataset_train__)):
            lgb_train = self.__dataset_train__[idx]
            lgb_valid = self.__dataset_valid__[idx]
            X_valid, y_valid = self.__validation__[idx]
            gbm = lgb.train( v_params,
                             lgb_train,
                             num_boost_round        = v_num_boost_round,
                             valid_sets             = [lgb_train, lgb_valid],
                             feature_name           = self.__feature_name__,
                             #categorical_feature=[21]
                             early_stopping_rounds  = 180,
                             verbose_eval           = p_verbose )
            v_fileName = f'models/saveTrain/model_{self.__modelName__}_{p_sufix}_{idx + 1}'
            gbm.save_model(f'{v_fileName}.txt', num_iteration = gbm.best_iteration)                
            if p_saveFull:                                
                with open(f'{v_fileName}.json', 'w+') as f:
                    json.dump(gbm.dump_model(), f, indent = 4)            
            bst = lgb.Booster(model_file = f'{v_fileName}.txt')  
            y_pred  = bst.predict(X_valid)
            v_score = roc_auc_score(y_valid, y_pred)
            self.__models__.append(bst)
            self.__validScores__.append(v_score)
        return
    
    
    #----------------------------------------------------------------------------------------------------------------
    def __scaleData__(self, X_data): 
        """ Returns a copy of the fead raw data. """ 
        return X_data.copy()
            
        
    #----------------------------------------------------------------------------------------------------------------
    def __createCVSplits__(self): 
        """ Function used to create the training / validation datasets. They will only be created one, at the 
            initialization of the objects. When using Bayesian Optimization, this operation is than executed only once. """ 
        if len(self.__dataset_train__) == 0: 
            self.__dataset_train__ = []
            self.__dataset_valid__ = []
            self.__validation__    = []
            self.__feature_name__  = self.__X_train__.columns.tolist()
            
            # For cross validation add the training / validation datasets based on the CVFold splits
            if self.__cvSplits__ > 1:
                print('Execute CV split.')
                v_cvFolds = CVFold(n_splits = self.__cvSplits__, shuffle = False, random_state = self.__random_state__)
                for train_idx, valid_idx in v_cvFolds.split(self.__X_train__, self.__y_train__):
                    X_trainCV = self.__scaleData__(self.__X_train__.iloc[train_idx, :], True)
                    y_trainCV = self.__y_train__.iloc[train_idx]
                    X_validCV = self.__scaleData__(self.__X_train__.iloc[valid_idx, :], False)
                    y_validCV = self.__y_train__.iloc[valid_idx]

                    if self.__modelType__ == 'lightgbm':
                        lgb_train = lgb.Dataset( X_trainCV, y_trainCV, free_raw_data = True )
                        lgb_valid = lgb.Dataset( X_validCV, y_validCV, free_raw_data = True, reference = lgb_train )
                        self.__dataset_train__.append(lgb_train)
                        self.__dataset_valid__.append(lgb_valid)   
                        self.__validation__.append((X_validCV, y_validCV))
            
            # Add the complete training dataset for LGB
            elif self.__modelType__ == 'lightgbm':
                lgb_train = lgb.Dataset( self.__scaleData__(self.__X_train__), 
                                         self.__y_train__, free_raw_data = True )
                lgb_valid = lgb.Dataset( self.__scaleData__(self.__X_train__), 
                                         self.__y_train__, free_raw_data = True, reference = lgb_train )
                self.__dataset_train__.append(lgb_train)
                self.__dataset_valid__.append(lgb_valid) 
                self.__validation__.append((self.__X_train__, self.__y_train__))
        
        return
    
    
    #----------------------------------------------------------------------------------------------------------------
    def bayesianSearchLGB( self, p_posWeight, p_max_eval, p_topTrials, X_test, y_test, p_verbose = False ):
        """ Execute the Bayesian Search for LGB models. It returns the best x model groups. """ 
        v_hyper_space = { 'objective':           'binary',
                          'n_estimators': 1000 + hp.randint('n_estimators', 1500),               
                          'boosting_type': hp.choice('boosting_type', [
                                                         { 'boosting_type': 'gbdt', 
                                                           'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                         { 'boosting_type': 'dart', 
                                                           'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                         { 'boosting_type': 'goss'} ]),
                          'max_depth':         hp.quniform('max_depth', 1, 30, 1),
                          'scale_pos_weight':  hp.quniform('scale_pos_weight', p_posWeight * 0.5, 
                                                                               p_posWeight * 2, 
                                                                               p_posWeight * 0.5),
                          'num_leaves':        hp.quniform('num_leaves', 2, 150, 1),
                          'learning_rate':     hp.loguniform('learning_rate', np.log(0.00001), np.log(0.2)),
                          'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
                          'reg_alpha':         hp.uniform('reg_alpha', 0.0, 1.0),
                          'reg_lambda':        hp.uniform('reg_lambda', 0.0, 1.0),
                          'colsample_bytree':  hp.uniform('colsample_by_tree', 0.6, 1.0) }   
        v_bayes_trials = Trials()
        
        self.ITERATION = 0
        self.__cvScores__ = {}
        import csv
        from hyperopt import STATUS_OK
        from timeit import default_timer as timer
        
        # File to save results
        v_fileName = f'models/gbm_trials/model_{self.__modelName__}.csv'
        v_fopen    = open(v_fileName, 'w')
        v_writer   = csv.writer(v_fopen)
        
        # Write the headers to the file
        v_writer.writerow([ 'scoreValid', 'scoreTest', 'loss', 'params', 'iteration', 'train_time' ])
        v_fopen.close()

        def objective(p_params, p_fileName = v_fileName):
            """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
            # Keep track of evals
            self.ITERATION += 1
            
            # Retrieve the subsample if present otherwise set to 1.0
            v_subsample = p_params['boosting_type'].get('subsample', 1.0)

            # Extract the boosting type
            p_params['boosting_type'] = p_params['boosting_type']['boosting_type']
            p_params['subsample']     = v_subsample
            
            # Make sure that the parameters which need to be integers are integers
            for parameter_name in ['num_leaves', 'min_child_samples', 'max_depth']:
                p_params[parameter_name] = int(p_params[parameter_name])

            v_start = timer()
            self.__trainLGB__( p_params          = p_params, 
                               p_sufix           = self.ITERATION,
                               p_saveFull        = False,
                               p_verbose         = False )
            v_run_time = timer() - v_start
            
            v_scoreValid = self.getScore()             
            v_predTest   = self.predictProba(X_test)
            v_scoreTest  = roc_auc_score(y_test, v_predTest)               
            self.__cvScores__[self.ITERATION] = { 'valid': v_scoreValid,
                                                  'test':  v_scoreTest }             
            if p_verbose: print(self.ITERATION, ' ... ', round(v_scoreValid, 6), ' ... ', round(v_scoreTest, 6))

            # Loss must be minimized
            v_loss = 1 - v_scoreValid
            
            # Write to the csv file
            v_fopen = open(p_fileName, 'a')
            v_writer = csv.writer(v_fopen)
            v_writer.writerow([v_scoreValid, v_scoreTest, v_loss, p_params, self.ITERATION, v_run_time])
            
            # Dictionary with information for evaluation
            return { 'loss':        v_loss, 
                     'params':      p_params, 
                     'status':      STATUS_OK, 
                     'iteration':   self.ITERATION,
                     'train_time':  v_run_time,
                     'scoreValid':  v_scoreValid,  
                     'scoreTest':   v_scoreTest }
        
        v_search = fmin( fn         = objective, 
                         space      = v_hyper_space, 
                         algo       = tpe.suggest, 
                         max_evals  = p_max_eval, 
                         trials     = v_bayes_trials )
        
        v_bayes_trials = sorted(v_bayes_trials.results, key=lambda x: x['loss'], reverse = False)
                                       
        print('\n***********************************************************************************************')
        v_src_files = os.listdir(f'models/gbm_trials/')
        v_now = datetime.datetime.today().strftime('%Y%m%d_%H%M')
        v_stackDir = f'models/stack/{self.__modelName__}_{v_now}'
        if not os.path.exists(v_stackDir):
            os.makedirs(v_stackDir)
        for item in v_bayes_trials[:p_topTrials]:
            print(item['iteration'], ' ... ', item['scoreValid'], ' ... ', item['scoreTest'])
            # Copy the models to the stack folder
            for idx in range(self.__cvSplits__):
                v_fileName = f'model_{self.__modelName__}_{item["iteration"]}_{idx + 1}'
                v_fileName = os.path.join(v_src_files, v_fileName)
                shutil.copy(v_fileName, v_stackDir)
        
        return v_bayes_trials