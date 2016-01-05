
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

        self.data['X_train'], self.data['X_probe'], self.data['y_train'], self.data['y_probe'] = skl_cv.train_test_split(X, y, test_size=0.20, random_state = 0)
    
    def generateModelPredictions(self, clf, param_grid):
        models_added = 0
        
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

        meta_learn.generateModelPredictions(lr_clf, lr_param_grid)


        # ## AdaBoost with DT stumps

        # In[7]:

        import sklearn.tree as skl_tree


        # In[10]:

        dt_stump_clf = skl_tree.DecisionTreeClassifier(max_depth=1)
        ada_dt_stump_clf = skl_ensemble.AdaBoostClassifier(base_estimator=dt_stump_clf)

        param_grid = [
            {'n_estimators':[50, 100, 500, 1000], 'learning_rate':[0.001, 0.1, 1.0] }
            ]

        ada_dt_stump_param_grid = skl_gs.ParameterGrid(param_grid = param_grid)

        meta_learn.generateModelPredictions(ada_dt_stump_clf, ada_dt_stump_param_grid)


        # In[11]:

        len(meta_learn.model_list)


        # ## Random Subspace Method with kNN

        # In[14]:

        import sklearn.neighbors as skl_neighbors


        # In[ ]:

        skl_ensemble.BaggingClassifier(skl_neighbors.)


    ...    