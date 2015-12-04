

import pandas as pd

# ### Predicting Unknowns
def predictAndExport(clf, X_test, test_ids):

    y_pred = clf.predict(X_test)
    
    test_output = pd.DataFrame([test_ids, y_pred]).T
    test_output.columns = ['PassengerId','Survived']
    test_output.head()

    test_output.to_csv('Data/test_output.csv', index=False)
