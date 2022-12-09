import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


KNN_pipeline = make_pipeline(StandardScaler(),
                             KNeighborsClassifier())
print(KNN_pipeline.get_params().keys())
parameters = {
    'kneighborsclassifier__n_neighbors' : [1,2,3,5,6],
    'kneighborsclassifier__metric' : ['eucledian', 'manhattan'],
    'kneighborsclassifier__weights' : ['uniform', 'distance']
    }

from sklearn.model_selection import GridSearchCV

CV_model = GridSearchCV(estimator=KNN_pipeline,
                        param_grid=parameters,
                        cv=10,
                        verbose=10,
                        n_jobs=-1,
                        scoring='roc_auc')





df = pd.read_json('fulldata.json')