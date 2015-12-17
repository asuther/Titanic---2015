
# coding: utf-8

# # Meta-Learning with predictor
import pandas as pd
import numpy as np
import sys
sys.path.append("C:\Users\Lundi\Documents\Programming\Python\Kaggle\Titanic - 2015")
sys.path.append("C:\Users\Lundi\Documents\Programming\Python\Kaggle\Titanic - 2015\Stacked Generalization")
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