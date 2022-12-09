import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score   

df = sns.load_dataset('tips')

df.pop('tip')

pd.options.display.max_rows = 100

print(df.head(10),'\n', '_'*25,  '\n', df.info(),'\n', '_'*25, '\n', df.describe())

col = df.columns

""" Factorize """  
def Factorize(d):
    dfac = d.copy()
    for i in col:
        if dfac[i].dtype.name == 'category':
            dfac[i] = pd.factorize(dfac[i])[0]
    d = dfac
    print(d.info())
    return d

""" Dummies """   
def Dummies(d):
    ddum = d.copy()
    for i in col:
        if ddum[i].dtype.name == 'category':
            k = pd.get_dummies(ddum[i], drop_first=True, dtype = int)
            ddum = pd.concat([ddum, k], axis = 1)
            del ddum[i]
    d = ddum
    print(d.info())
    return d

""" Dummies: do no drop first! """  
def DummiesNoDropFirst(d):
    ddum = d.copy()     
    for i in col:
        if ddum[i].dtype.name == 'category':
            k = pd.get_dummies(ddum[i], dtype = int)
            ddum = pd.concat([ddum, k], axis = 1)
            del ddum[i]
    d = ddum
    print(d.info())
    return d

""" RFC """
def Analysis(data):
       
    y = data['size']
    data.pop('size')
    X = data
      
    print(X.info(), y.info())
      
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
        
    classifier = RandomForestClassifier(n_estimators=1, criterion="entropy" ,random_state=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
        
    # Оценка
    print('---------------- Оценка ---------------------')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

Analysis(Dummies(df))























