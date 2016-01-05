import sklearn.linear_model as skl_lm
import pandas as pd
import numpy as np

def __init__(self):
    import socket
    computer_name = socket.gethostname()
    if computer_name == 'Alexs-MacBook-Pro.local':
        os.chdir("/Users/alexsutherland/Documents/Programming/Python/Kaggle/Titanic---2015")
    else:    
        os.chdir('C:\Users\Lundi\Documents\Programming\Python\Kaggle\Titanic - 2015')
    
    get_ipython().magic(u'matplotlib inline')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 12,7

    
    import sklearn.cross_validation as skl_cv
    import sklearn.preprocessing as skl_pre
    from sklearn.grid_search import GridSearchCV
    
    # ### Loading the data


    titanic_data_v5 = pd.read_csv('Data/titanic_data_v5.csv')
    titanic_data_v5 = titanic_data_v5.drop(['Unnamed: 0'], axis=1)
    titanic_data_v5.head(1)

    test_data = pd.read_csv('Data/test.csv')

    # ### Building the optimum model

    X = titanic_data_v5.drop(['PassengerId','Survived'], axis=1)
    y = titanic_data_v5['Survived']
    
    scaler = skl_pre.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    lr_clf = skl_lm.LogisticRegression()

    gs_params = [
        {'penalty': ['l1'], 'C': np.logspace(-4,4, num=50)},
        {'penalty': ['l2'], 'C': np.logspace(-4,4, num=50)}
    ]

    gs_lr_clf = GridSearchCV(lr_clf, param_grid = gs_params, cv=10)
    gs_lr_clf.fit(X_scaled, y);
    best_lr_clf = gs_lr_clf.best_estimator_

    
def getData():
    #Get computer name
    import os
    import socket
    computer_name = socket.gethostname()
    if computer_name == 'Alexs-MacBook-Pro.local':
        base_path = "/Users/alexsutherland/Documents/Programming/Python/Kaggle/Titanic---2015"
    else:    
        base_path = 'C:\Users\Lundi\Documents\Programming\Python\Kaggle\Titanic - 2015'
    
    
    os.chdir(base_path)
    #Training Data 
    import pandas as pd
    titanic_data_v5 = pd.read_csv('Data/titanic_data_v5.csv')
    titanic_data_v5 = titanic_data_v5.drop(['Unnamed: 0'], axis=1)
    titanic_data_v5.head(1)

    X_train = titanic_data_v5.drop(['PassengerId','Survived'], axis=1)
    y_train = titanic_data_v5['Survived']
    
    #Test Data
    test_data = pd.read_csv('Data/test.csv')
    X_test_ids = test_data['PassengerId']
    X_test = preprocessTestData(test_data)
    
    
    return X_train, y_train, X_test, X_test_ids.values
# ### Imputating ages with linear regression

# In[65]:

def getMissingAges(data):
    X = data.dropna()
    y = X['Age']
    X = X.drop(['Age'], axis=1)
    X.head(2)
    
    lr_reg = skl_lm.LinearRegression()
    
    lr_reg.fit(X, y)
    
    #Get missing age data
    X_without_age = data.ix[np.isnan(data['Age']),:].drop(['Age'], axis=1)
    X_without_age.head(2)
    
    #Predict
    age_predict = lr_reg.predict(X_without_age)
    
    #Replace any negative predictions with 0.5
    age_predict[age_predict < 0] = 0.5
    
    data_copy = data.copy()
    data_copy.ix[np.isnan(data_copy['Age']),'Age'] = age_predict
    return data_copy


# ### Preprocessing Test Data

# In[64]:
def preprocessTestData(test_data):
    import pandas as pd
    X_test = test_data.drop(['PassengerId', 'Name','Ticket','Cabin'], axis=1)

    #Sex
    sex_dummies = pd.get_dummies(X_test['Sex'])
    X_test['Is_male'] = sex_dummies['male']
    X_test = X_test.drop(['Sex'], axis=1)

    #Embarked
    X_test[['Embarked Q','Embarked S']] = pd.get_dummies(X_test['Embarked']).drop(['C'],axis=1)
    X_test = X_test.drop(['Embarked'], axis=1)
    

    X_test_full_ages = getMissingAges(X_test)


    # There's one fare entry that is missing, let's insert the average fare for people in his same class:

    pclasses_of_missing_fares = X_test_full_ages.ix[np.isnan(X_test_full_ages['Fare']),'Pclass']
    mean_fare_for_pclass = X_test_full_ages.ix[X_test_full_ages['Pclass'] == int(pclasses_of_missing_fares.values), 'Fare'].mean()

    X_test_full_ages['Fare'] = X_test_full_ages['Fare'].fillna(mean_fare_for_pclass)
    
    return X_test_full_ages





# ### Predicting Unknowns
def predictAndExport(X_test, test_ids):

    y_pred = best_lr_clf.predict(X_test)
    
    test_output = pd.DataFrame([test_ids.values, y_pred]).T
    test_output.columns = ['PassengerId','Survived']
    test_output.head()

    test_output.to_csv('Data/test_output.csv', index=False)

