
# coding: utf-8

# # Meta-Learning with predictor
import pandas as pd
import numpy as np
import sys
import socket
computer_name = socket.gethostname()
if computer_name == 'Alexs-MacBook-Pro.local':
    base_path = "/Users/alexsutherland/Documents/Programming/Python/Kaggle/Titanic---2015"
else:    
    base_path = 'C:\Users\Lundi\Documents\Programming\Python\Kaggle\Titanic - 2015'
sys.path.append(base_path)
sys.path.append(base_path + "\Stacked Generalization")
import TitanicPreprocessor as tp
import TitanicPredictor as tpred
import sklearn.cross_validation as skl_cv

class metaLearning():
    
    def __init__(self):
        self.data = {'X_train':[], 'X_probe':[], 'y_train':[], 'y_probe': [], 'X_test':[], 'X_test_ids':[]}
        self.model_list = []
        self.model_prediction_probs = []
        
        X, y, self.data['X_test'], self.data['X_test_ids'] = tp.getData()
        self.X = X
        self.y = y
        self.data['X_train'], self.data['X_probe'], self.data['y_train'], self.data['y_probe'] = skl_cv.train_test_split(X, y, test_size=0.40, random_state = 0)
    
    def generateModelPredictions(self, clf, param_grid):
        models_added = 0
        
        n_folds = 10
        skf = list(skl_cv.StratifiedKFold(self.data['y'], n_folds))
        
        self.dataset_blend_train = np.zeros((self.data['X'].shape[0], len(param_grid)))
        self.dataset_blend_test = np.zeros((self.data['X_test'].shape[0], len(param_grid)))

        for current_params in param_grid:
            print current_params
            dataset_blend_test_j = np.zeros((self.data['X_test'].shape[0], len(skf)))
            for i, (train, test) in enumerate(skf):
                print "Fold", i
                X_train = X.ix[train,:]
                y_train = y.ix[train]
                X_test = X.ix[test,:]
                y_test = y.ix[test]
                
                clf.set_params(**current_params)
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:,1]
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(self.data['X_test'])[:,1]
            dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        
        
        
        
        for current_params in param_grid:
            #Set Params for model
            clf.set_params(**current_params)
            #Train
            X_train = self.data['X_train']
            y_train = self.data['y_train']
            clf.fit(X_train, y_train)
            
            #Predict
            current_pred_probs = clf.predict_proba(self.data['X_probe'])
            self.model_prediction_probs.append(current_pred_probs.transpose()[1])
            self.model_list.append(str(clf))
            models_added+=1
        print models_added, 'models added'
    def generateSingleModelPredictions(self, clf):
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        clf.fit(X_train, y_train)
            
        #Predict
        current_pred_probs = clf.predict_proba(self.data['X_probe'])
        self.model_prediction_probs.append(current_pred_probs.transpose()[1])
        self.model_list.append(str(clf))
        
    def generateExampleModels(self):
        import sklearn.ensemble as skl_ensemble
        import sklearn.linear_model as skl_lm
        import sklearn.grid_search as skl_gs
        import sklearn.cross_validation as skl_cv


        # ## Random Forest
        
        rf_clf = skl_ensemble.RandomForestClassifier()

        param_grid = [
            {'n_estimators': [10,100, 1000], 'criterion':['gini','entropy'], 'max_depth':[None,1,2,4,6,8,10], 'max_features':[1,2,4] }
            ]

        rf_param_grid = skl_gs.ParameterGrid(param_grid = param_grid)

        self.generateModelPredictions(rf_clf, rf_param_grid)
        
        # ## Logistic Regression

        lr_clf = skl_lm.LogisticRegression()

        param_grid = [
            {'penalty':['l1','l2'], 'C':np.logspace(-10,10,num=50), 'solver': ['liblinear'] }
            ]

        lr_param_grid = skl_gs.ParameterGrid(param_grid = param_grid)

        self.generateModelPredictions(lr_clf, lr_param_grid)


        # ## AdaBoost with DT stumps

        # In[7]:

        import sklearn.tree as skl_tree

        dt_stump_clf = skl_tree.DecisionTreeClassifier(max_depth=1)
        ada_dt_stump_clf = skl_ensemble.AdaBoostClassifier(base_estimator=dt_stump_clf)

        param_grid = [
            {'n_estimators':[50, 100, 500, 1000], 'learning_rate':[0.001, 0.1, 1.0] }
            ]

        ada_dt_stump_param_grid = skl_gs.ParameterGrid(param_grid = param_grid)

        self.generateModelPredictions(ada_dt_stump_clf, ada_dt_stump_param_grid)

        len(self.model_list)
    def generateExampleLogisticModels(self):
        import sklearn.ensemble as skl_ensemble
        import sklearn.linear_model as skl_lm
        import sklearn.grid_search as skl_gs
        import sklearn.cross_validation as skl_cv

        # ## Logistic Regression

        lr_clf = skl_lm.LogisticRegression()

        param_grid = [
            {'penalty':['l1','l2'], 'C':np.logspace(-10,10,num=50), 'solver': ['liblinear'] }
            ]

        lr_param_grid = skl_gs.ParameterGrid(param_grid = param_grid)

        self.generateModelPredictions(lr_clf, lr_param_grid)
